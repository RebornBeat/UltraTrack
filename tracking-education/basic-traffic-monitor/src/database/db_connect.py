"""
Database connection module for managing database interactions.
Provides connection pooling, query execution, and transaction management.
"""

import os
import time
import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor, execute_values

# Configure logger for this module
logger = logging.getLogger(__name__)


@dataclass
class ConnectionParams:
    """Database connection parameters."""
    host: str
    port: int
    database: str
    user: str
    password: str


class DatabaseConnection:
    """
    Database connection manager using connection pooling.
    """
    
    def __init__(
        self,
        connection_params: ConnectionParams,
        min_connections: int = 1,
        max_connections: int = 10,
        connection_timeout: int = 30,
        retry_attempts: int = 3,
        retry_delay: int = 2
    ):
        """
        Initialize database connection manager.
        
        Args:
            connection_params: Database connection parameters
            min_connections: Minimum number of connections in pool
            max_connections: Maximum number of connections in pool
            connection_timeout: Connection timeout in seconds
            retry_attempts: Number of retry attempts for failed operations
            retry_delay: Delay between retry attempts in seconds
        """
        self.params = connection_params
        self.min_connections = min_connections
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        # Connection pool
        self.pool = None
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Stats
        self.connection_count = 0
        self.query_count = 0
        self.error_count = 0
        
        # Initialize connection pool
        self._initialize_pool()
        
        logger.info(f"Database connection manager initialized for {connection_params.host}:{connection_params.port}/{connection_params.database}")
    
    def _initialize_pool(self):
        """Initialize the connection pool."""
        try:
            self.pool = pool.ThreadedConnectionPool(
                self.min_connections,
                self.max_connections,
                host=self.params.host,
                port=self.params.port,
                database=self.params.database,
                user=self.params.user,
                password=self.params.password,
                connect_timeout=self.connection_timeout
            )
            
            logger.info("Connection pool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {str(e)}")
            self.pool = None
            self.error_count += 1
            raise
    
    def get_connection(self) -> Tuple[Any, int]:
        """
        Get a connection from the pool.
        
        Returns:
            Tuple of (connection, connection_id)
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")
        
        try:
            with self.lock:
                conn = self.pool.getconn()
                self.connection_count += 1
                conn_id = self.connection_count
            
            logger.debug(f"Got connection {conn_id} from pool")
            return conn, conn_id
        except Exception as e:
            logger.error(f"Failed to get connection from pool: {str(e)}")
            self.error_count += 1
            raise
    
    def return_connection(self, conn, conn_id: int):
        """
        Return a connection to the pool.
        
        Args:
            conn: Connection object
            conn_id: Connection ID
        """
        if not self.pool:
            return
        
        try:
            self.pool.putconn(conn)
            logger.debug(f"Returned connection {conn_id} to pool")
        except Exception as e:
            logger.error(f"Failed to return connection {conn_id} to pool: {str(e)}")
            self.error_count += 1
    
    def execute_query(
        self,
        query: str,
        params: Optional[Union[Tuple, Dict[str, Any]]] = None,
        fetchall: bool = True,
        dict_cursor: bool = True
    ) -> Union[List[Dict[str, Any]], List[Tuple], None]:
        """
        Execute a database query with retry logic.
        
        Args:
            query: SQL query string
            params: Query parameters
            fetchall: Whether to fetch all results
            dict_cursor: Whether to use dictionary cursor
        
        Returns:
            Query results or None
        """
        conn = None
        conn_id = -1
        
        for attempt in range(self.retry_attempts):
            try:
                conn, conn_id = self.get_connection()
                
                # Use dictionary cursor if requested
                cursor_factory = RealDictCursor if dict_cursor else None
                with conn.cursor(cursor_factory=cursor_factory) as cursor:
                    # Execute query
                    cursor.execute(query, params)
                    
                    # Fetch results if needed
                    if cursor.description and fetchall:
                        result = cursor.fetchall()
                    elif cursor.description:
                        result = cursor.fetchone()
                    else:
                        result = None
                    
                    # Commit transaction
                    conn.commit()
                
                with self.lock:
                    self.query_count += 1
                
                return result
            
            except Exception as e:
                logger.error(f"Query execution failed (attempt {attempt+1}/{self.retry_attempts}): {str(e)}")
                
                if conn:
                    try:
                        conn.rollback()
                    except:
                        pass
                
                with self.lock:
                    self.error_count += 1
                
                # Last attempt, re-raise the exception
                if attempt == self.retry_attempts - 1:
                    raise
                
                # Wait before retrying
                time.sleep(self.retry_delay)
            
            finally:
                # Return connection to pool
                if conn:
                    self.return_connection(conn, conn_id)
    
    def execute_transaction(
        self,
        queries: List[Tuple[str, Optional[Union[Tuple, Dict[str, Any]]]]],
        dict_cursor: bool = True
    ) -> bool:
        """
        Execute a batch of queries in a single transaction.
        
        Args:
            queries: List of (query, params) tuples
            dict_cursor: Whether to use dictionary cursor
        
        Returns:
            True if transaction succeeded, False otherwise
        """
        conn = None
        conn_id = -1
        
        for attempt in range(self.retry_attempts):
            try:
                conn, conn_id = self.get_connection()
                
                # Use dictionary cursor if requested
                cursor_factory = RealDictCursor if dict_cursor else None
                with conn.cursor(cursor_factory=cursor_factory) as cursor:
                    # Execute each query in the transaction
                    for query, params in queries:
                        cursor.execute(query, params)
                    
                    # Commit transaction
                    conn.commit()
                
                with self.lock:
                    self.query_count += len(queries)
                
                return True
            
            except Exception as e:
                logger.error(f"Transaction execution failed (attempt {attempt+1}/{self.retry_attempts}): {str(e)}")
                
                if conn:
                    try:
                        conn.rollback()
                    except:
                        pass
                
                with self.lock:
                    self.error_count += 1
                
                # Last attempt, re-raise the exception
                if attempt == self.retry_attempts - 1:
                    raise
                
                # Wait before retrying
                time.sleep(self.retry_delay)
            
            finally:
                # Return connection to pool
                if conn:
                    self.return_connection(conn, conn_id)
        
        return False
    
    def execute_batch_insert(
        self,
        table_name: str,
        columns: List[str],
        values: List[Tuple],
        page_size: int = 1000
    ) -> int:
        """
        Execute a batch insert operation.
        
        Args:
            table_name: Name of the table
            columns: List of column names
            values: List of value tuples
            page_size: Number of rows to insert in each batch
        
        Returns:
            Number of rows inserted
        """
        if not values:
            return 0
        
        conn = None
        conn_id = -1
        total_inserted = 0
        
        try:
            conn, conn_id = self.get_connection()
            
            # Create the insert query
            column_str = ', '.join(columns)
            query = f"INSERT INTO {table_name} ({column_str}) VALUES %s"
            
            # Split values into batches
            for i in range(0, len(values), page_size):
                batch = values[i:i+page_size]
                
                with conn.cursor() as cursor:
                    # Use execute_values for efficient batch insert
                    execute_values(cursor, query, batch)
                
                # Commit each batch
                conn.commit()
                total_inserted += len(batch)
                
                logger.debug(f"Inserted batch of {len(batch)} rows into {table_name}")
            
            with self.lock:
                self.query_count += (total_inserted + page_size - 1) // page_size
            
            return total_inserted
            
        except Exception as e:
            logger.error(f"Batch insert failed: {str(e)}")
            
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            
            with self.lock:
                self.error_count += 1
            
            raise
            
        finally:
            # Return connection to pool
            if conn:
                self.return_connection(conn, conn_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database connection statistics.
        
        Returns:
            Dictionary of statistics
        """
        with self.lock:
            stats = {
                'connection_count': self.connection_count,
                'query_count': self.query_count,
                'error_count': self.error_count
            }
            
            if self.pool:
                try:
                    stats['pool_used'] = len(self.pool._used)
                    stats['pool_free'] = len(self.pool._pool)
                except:
                    pass
            
            return stats
    
    def close(self):
        """Close all connections and the connection pool."""
        if self.pool:
            self.pool.closeall()
            logger.info("Closed all database connections")
            self.pool = None
    
    def __del__(self):
        """Ensure resources are cleaned up on deletion."""
        self.close()


# Module-level connection instance
_db_connection = None


def init_db_connection(
    connection_params: ConnectionParams,
    min_connections: int = 1,
    max_connections: int = 10
) -> DatabaseConnection:
    """
    Initialize the module-level database connection.
    
    Args:
        connection_params: Database connection parameters
        min_connections: Minimum number of connections in pool
        max_connections: Maximum number of connections in pool
    
    Returns:
        Database connection instance
    """
    global _db_connection
    
    if _db_connection:
        _db_connection.close()
    
    _db_connection = DatabaseConnection(
        connection_params,
        min_connections=min_connections,
        max_connections=max_connections
    )
    
    return _db_connection


def get_db_connection() -> DatabaseConnection:
    """
    Get the module-level database connection.
    
    Returns:
        Database connection instance
    """
    global _db_connection
    
    if not _db_connection:
        raise RuntimeError("Database connection not initialized")
    
    return _db_connection


def execute_query(
    query: str,
    params: Optional[Union[Tuple, Dict[str, Any]]] = None,
    fetchall: bool = True,
    dict_cursor: bool = True
) -> Union[List[Dict[str, Any]], List[Tuple], None]:
    """
    Execute a database query using the module-level connection.
    
    Args:
        query: SQL query string
        params: Query parameters
        fetchall: Whether to fetch all results
        dict_cursor: Whether to use dictionary cursor
    
    Returns:
        Query results or None
    """
    db = get_db_connection()
    return db.execute_query(query, params, fetchall, dict_cursor)


def execute_transaction(
    queries: List[Tuple[str, Optional[Union[Tuple, Dict[str, Any]]]]],
    dict_cursor: bool = True
) -> bool:
    """
    Execute a batch of queries in a single transaction using the module-level connection.
    
    Args:
        queries: List of (query, params) tuples
        dict_cursor: Whether to use dictionary cursor
    
    Returns:
        True if transaction succeeded, False otherwise
    """
    db = get_db_connection()
    return db.execute_transaction(queries, dict_cursor)
