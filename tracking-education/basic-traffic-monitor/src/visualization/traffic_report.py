"""
Traffic report generation for creating visualizations and reports from traffic data.
Provides various report types and export formats for traffic analysis results.
"""

import os
import json
import time
import logging
import datetime
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Configure logger for this module
logger = logging.getLogger(__name__)

# Use Agg backend for matplotlib (non-interactive, thread-safe)
matplotlib.use('Agg')


class ReportType(Enum):
    """Type of traffic report."""
    VEHICLE_COUNT = 'vehicle_count'
    SPEED_DISTRIBUTION = 'speed_distribution'
    TRAFFIC_FLOW = 'traffic_flow'
    HOURLY_DISTRIBUTION = 'hourly_distribution'
    VEHICLE_TYPES = 'vehicle_types'
    LICENSE_PLATES = 'license_plates'
    CUSTOM = 'custom'


class ChartType(Enum):
    """Type of chart for visualization."""
    LINE = 'line'
    BAR = 'bar'
    PIE = 'pie'
    SCATTER = 'scatter'
    HEATMAP = 'heatmap'
    HISTOGRAM = 'histogram'
    TABLE = 'table'


class ExportFormat(Enum):
    """Export format for reports."""
    PNG = 'png'
    JPG = 'jpg'
    SVG = 'svg'
    PDF = 'pdf'
    CSV = 'csv'
    JSON = 'json'
    HTML = 'html'


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    title: str
    subtitle: Optional[str] = None
    chart_type: ChartType = ChartType.BAR
    width: int = 1200
    height: int = 800
    dpi: int = 100
    colors: Optional[List[str]] = None
    grid: bool = True
    legend: bool = True
    date_format: str = '%Y-%m-%d %H:%M'
    export_format: ExportFormat = ExportFormat.PNG
    include_timestamp: bool = True
    include_logo: bool = False
    logo_path: Optional[str] = None
    custom_style: Optional[Dict[str, Any]] = None


class ReportGenerator:
    """
    Generate traffic reports and visualizations from traffic data.
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """
        Initialize report generator.
        
        Args:
            config: Optional default report configuration
        """
        self.config = config or ReportConfig(title="Traffic Report")
        
        # Set default matplotlib style
        plt.style.use('ggplot')
        
        logger.info("Report generator initialized")
    
    def generate_report(
        self,
        report_type: ReportType,
        data: Any,
        config: Optional[ReportConfig] = None,
        output_path: Optional[str] = None
    ) -> Union[str, bytes, Figure]:
        """
        Generate a traffic report.
        
        Args:
            report_type: Type of report to generate
            data: Data for the report
            config: Optional report configuration (overrides default)
            output_path: Optional path to save the report
        
        Returns:
            Path to saved file, bytes data, or matplotlib Figure object
        """
        # Use provided config or default
        report_config = config or self.config
        
        # Generate the appropriate report
        if report_type == ReportType.VEHICLE_COUNT:
            result = self._generate_vehicle_count_report(data, report_config)
        elif report_type == ReportType.SPEED_DISTRIBUTION:
            result = self._generate_speed_distribution_report(data, report_config)
        elif report_type == ReportType.TRAFFIC_FLOW:
            result = self._generate_traffic_flow_report(data, report_config)
        elif report_type == ReportType.HOURLY_DISTRIBUTION:
            result = self._generate_hourly_distribution_report(data, report_config)
        elif report_type == ReportType.VEHICLE_TYPES:
            result = self._generate_vehicle_types_report(data, report_config)
        elif report_type == ReportType.LICENSE_PLATES:
            result = self._generate_license_plates_report(data, report_config)
        elif report_type == ReportType.CUSTOM:
            if not callable(data):
                raise ValueError("For CUSTOM report type, data must be a callable function")
            result = data(report_config)
        else:
            raise ValueError(f"Unsupported report type: {report_type}")
        
        # If a figure was returned and output path specified, save it
        if isinstance(result, Figure) and output_path:
            result.savefig(output_path, dpi=report_config.dpi, bbox_inches='tight')
            plt.close(result)
            return output_path
        
        return result
    
    def _generate_vehicle_count_report(self, data: Dict[str, Any], config: ReportConfig) -> Figure:
        """
        Generate vehicle count report.
        
        Args:
            data: Dictionary with 'timestamps' and 'counts' lists
            config: Report configuration
        
        Returns:
            Matplotlib Figure object
        """
        # Extract data
        timestamps = data.get('timestamps', [])
        counts = data.get('counts', [])
        
        if not timestamps or not counts or len(timestamps) != len(counts):
            raise ValueError("Invalid data format for vehicle count report")
        
        # Convert timestamps to datetime if they are numeric
        if isinstance(timestamps[0], (int, float)):
            timestamps = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
        
        # Create figure
        fig = plt.figure(figsize=(config.width/config.dpi, config.height/config.dpi), dpi=config.dpi)
        ax = fig.add_subplot(111)
        
        # Plot data
        if config.chart_type == ChartType.LINE:
            ax.plot(timestamps, counts, marker='o', linestyle='-', 
                   color=config.colors[0] if config.colors else 'b')
        elif config.chart_type == ChartType.BAR:
            ax.bar(timestamps, counts, 
                 color=config.colors[0] if config.colors else 'b',
                 width=0.8 * (timestamps[1] - timestamps[0]).total_seconds() if len(timestamps) > 1 else 0.8)
        else:
            # Default to line chart for other types
            ax.plot(timestamps, counts, marker='o', linestyle='-', 
                   color=config.colors[0] if config.colors else 'b')
        
        # Configure axis and labels
        ax.set_xlabel('Time')
        ax.set_ylabel('Vehicle Count')
        ax.set_title(config.title)
        
        if config.subtitle:
            fig.suptitle(config.subtitle, fontsize=10)
        
        # Format x-axis as dates
        fig.autofmt_xdate()
        
        # Add grid if requested
        ax.grid(config.grid)
        
        # Add timestamp if requested
        if config.include_timestamp:
            fig.text(0.99, 0.01, f"Generated: {datetime.datetime.now().strftime(config.date_format)}",
                    ha='right', va='bottom', fontsize=8, color='gray')
        
        # Add logo if requested
        if config.include_logo and config.logo_path and os.path.exists(config.logo_path):
            try:
                logo = plt.imread(config.logo_path)
                ax_logo = fig.add_axes([0.01, 0.01, 0.1, 0.1], frameon=False)
                ax_logo.imshow(logo)
                ax_logo.axis('off')
            except Exception as e:
                logger.warning(f"Failed to add logo to report: {str(e)}")
        
        # Apply custom style if provided
        if config.custom_style:
            for key, value in config.custom_style.items():
                try:
                    if hasattr(ax, key):
                        getattr(ax, key)(value)
                    elif key.startswith('set_'):
                        getattr(ax, key)(value)
                except Exception as e:
                    logger.warning(f"Failed to apply custom style '{key}': {str(e)}")
        
        fig.tight_layout()
        return fig
    
    def _generate_speed_distribution_report(self, data: Dict[str, Any], config: ReportConfig) -> Figure:
        """
        Generate speed distribution report.
        
        Args:
            data: Dictionary with 'speeds' list and optional 'bins' count
            config: Report configuration
        
        Returns:
            Matplotlib Figure object
        """
        # Extract data
        speeds = data.get('speeds', [])
        bins = data.get('bins', 10)
        
        if not speeds:
            raise ValueError("Invalid data format for speed distribution report")
        
        # Create figure
        fig = plt.figure(figsize=(config.width/config.dpi, config.height/config.dpi), dpi=config.dpi)
        ax = fig.add_subplot(111)
        
        # Plot data
        if config.chart_type == ChartType.HISTOGRAM:
            ax.hist(speeds, bins=bins, color=config.colors[0] if config.colors else 'b')
        elif config.chart_type == ChartType.BAR:
            # Create histogram data manually
            hist, bin_edges = np.histogram(speeds, bins=bins)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ax.bar(bin_centers, hist, width=bin_edges[1] - bin_edges[0], 
                 color=config.colors[0] if config.colors else 'b')
        else:
            # Default to histogram
            ax.hist(speeds, bins=bins, color=config.colors[0] if config.colors else 'b')
        
        # Configure axis and labels
        ax.set_xlabel('Speed (km/h)')
        ax.set_ylabel('Count')
        ax.set_title(config.title)
        
        if config.subtitle:
            fig.suptitle(config.subtitle, fontsize=10)
        
        # Add grid if requested
        ax.grid(config.grid)
        
        # Add statistics annotations
        mean_speed = np.mean(speeds)
        median_speed = np.median(speeds)
        max_speed = np.max(speeds)
        min_speed = np.min(speeds)
        
        stats_text = (f"Mean: {mean_speed:.1f} km/h\n"
                     f"Median: {median_speed:.1f} km/h\n"
                     f"Max: {max_speed:.1f} km/h\n"
                     f"Min: {min_speed:.1f} km/h")
        
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
              fontsize=9, va='top', ha='right',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add timestamp if requested
        if config.include_timestamp:
            fig.text(0.99, 0.01, f"Generated: {datetime.datetime.now().strftime(config.date_format)}",
                    ha='right', va='bottom', fontsize=8, color='gray')
        
        # Add logo if requested
        if config.include_logo and config.logo_path and os.path.exists(config.logo_path):
            try:
                logo = plt.imread(config.logo_path)
                ax_logo = fig.add_axes([0.01, 0.01, 0.1, 0.1], frameon=False)
                ax_logo.imshow(logo)
                ax_logo.axis('off')
            except Exception as e:
                logger.warning(f"Failed to add logo to report: {str(e)}")
        
        # Apply custom style if provided
        if config.custom_style:
            for key, value in config.custom_style.items():
                try:
                    if hasattr(ax, key):
                        getattr(ax, key)(value)
                    elif key.startswith('set_'):
                        getattr(ax, key)(value)
                except Exception as e:
                    logger.warning(f"Failed to apply custom style '{key}': {str(e)}")
        
        fig.tight_layout()
        return fig
    
    def _generate_traffic_flow_report(self, data: Dict[str, Any], config: ReportConfig) -> Figure:
        """
        Generate traffic flow report.
        
        Args:
            data: Dictionary with 'timestamps', 'regions', and 'densities' lists
            config: Report configuration
        
        Returns:
            Matplotlib Figure object
        """
        # Extract data
        timestamps = data.get('timestamps', [])
        regions = data.get('regions', [])
        densities = data.get('densities', [])
        
        if not timestamps or not regions or not densities:
            raise ValueError("Invalid data format for traffic flow report")
        
        # Convert timestamps to datetime if they are numeric
        if isinstance(timestamps[0], (int, float)):
            timestamps = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
        
        # Create figure
        fig = plt.figure(figsize=(config.width/config.dpi, config.height/config.dpi), dpi=config.dpi)
        ax = fig.add_subplot(111)
        
        # Plot data based on chart type
        if config.chart_type == ChartType.LINE:
            for i, region in enumerate(regions):
                color = config.colors[i % len(config.colors)] if config.colors else None
                ax.plot(timestamps, [d[i] for d in densities], 
                       marker='o', linestyle='-', label=region, color=color)
        
        elif config.chart_type == ChartType.HEATMAP:
            # Create a 2D array of densities
            density_array = np.array(densities)
            im = ax.imshow(density_array.T, aspect='auto', cmap='viridis')
            
            # Set ticks and labels
            ax.set_xticks(np.arange(len(timestamps)))
            ax.set_yticks(np.arange(len(regions)))
            
            # Format x-axis labels based on timestamps
            time_labels = [ts.strftime('%H:%M') for ts in timestamps]
            ax.set_xticklabels(time_labels)
            ax.set_yticklabels(regions)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Density')
            
        else:
            # Default to line chart
            for i, region in enumerate(regions):
                color = config.colors[i % len(config.colors)] if config.colors else None
                ax.plot(timestamps, [d[i] for d in densities], 
                       marker='o', linestyle='-', label=region, color=color)
        
        # Configure axis and labels
        ax.set_xlabel('Time')
        ax.set_ylabel('Traffic Density')
        ax.set_title(config.title)
        
        if config.subtitle:
            fig.suptitle(config.subtitle, fontsize=10)
        
        # Add legend if requested and not heatmap
        if config.legend and config.chart_type != ChartType.HEATMAP:
            ax.legend()
        
        # Format x-axis as dates for non-heatmap charts
        if config.chart_type != ChartType.HEATMAP:
            fig.autofmt_xdate()
        
        # Add grid if requested
        ax.grid(config.grid)
        
        # Add timestamp if requested
        if config.include_timestamp:
            fig.text(0.99, 0.01, f"Generated: {datetime.datetime.now().strftime(config.date_format)}",
                    ha='right', va='bottom', fontsize=8, color='gray')
        
        # Add logo if requested
        if config.include_logo and config.logo_path and os.path.exists(config.logo_path):
            try:
                logo = plt.imread(config.logo_path)
                ax_logo = fig.add_axes([0.01, 0.01, 0.1, 0.1], frameon=False)
                ax_logo.imshow(logo)
                ax_logo.axis('off')
            except Exception as e:
                logger.warning(f"Failed to add logo to report: {str(e)}")
        
        # Apply custom style if provided
        if config.custom_style:
            for key, value in config.custom_style.items():
                try:
                    if hasattr(ax, key):
                        getattr(ax, key)(value)
                    elif key.startswith('set_'):
                        getattr(ax, key)(value)
                except Exception as e:
                    logger.warning(f"Failed to apply custom style '{key}': {str(e)}")
        
        fig.tight_layout()
        return fig
    
    def _generate_hourly_distribution_report(self, data: Dict[str, Any], config: ReportConfig) -> Figure:
        """
        Generate hourly distribution report.
        
        Args:
            data: Dictionary with 'hours' and 'counts' lists
            config: Report configuration
        
        Returns:
            Matplotlib Figure object
        """
        # Extract data
        hours = data.get('hours', list(range(24)))
        counts = data.get('counts', [0] * 24)
        
        if len(hours) != len(counts):
            raise ValueError("Hours and counts must have the same length")
        
        # Create figure
        fig = plt.figure(figsize=(config.width/config.dpi, config.height/config.dpi), dpi=config.dpi)
        ax = fig.add_subplot(111)
        
        # Plot data
        if config.chart_type == ChartType.BAR:
            ax.bar(hours, counts, color=config.colors[0] if config.colors else 'b')
        elif config.chart_type == ChartType.LINE:
            ax.plot(hours, counts, marker='o', linestyle='-', 
                   color=config.colors[0] if config.colors else 'b')
        else:
            # Default to bar chart
            ax.bar(hours, counts, color=config.colors[0] if config.colors else 'b')
        
        # Configure axis and labels
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Count')
        ax.set_title(config.title)
        
        if config.subtitle:
            fig.suptitle(config.subtitle, fontsize=10)
        
        # Set x-axis to show all hours
        ax.set_xticks(range(24))
        ax.set_xlim(-0.5, 23.5)
        
        # Add grid if requested
        ax.grid(config.grid)
        
        # Add annotations for peak hours
        peak_hour = hours[np.argmax(counts)]
        peak_count = max(counts)
        
        ax.annotate(f"Peak: {peak_hour}:00 ({peak_count})",
                  xy=(peak_hour, peak_count),
                  xytext=(peak_hour, peak_count + max(counts) * 0.1),
                  arrowprops=dict(facecolor='black', shrink=0.05),
                  ha='center')
        
        # Add timestamp if requested
        if config.include_timestamp:
            fig.text(0.99, 0.01, f"Generated: {datetime.datetime.now().strftime(config.date_format)}",
                    ha='right', va='bottom', fontsize=8, color='gray')
        
        # Add logo if requested
        if config.include_logo and config.logo_path and os.path.exists(config.logo_path):
            try:
                logo = plt.imread(config.logo_path)
                ax_logo = fig.add_axes([0.01, 0.01, 0.1, 0.1], frameon=False)
                ax_logo.imshow(logo)
                ax_logo.axis('off')
            except Exception as e:
                logger.warning(f"Failed to add logo to report: {str(e)}")
        
        # Apply custom style if provided
        if config.custom_style:
            for key, value in config.custom_style.items():
                try:
                    if hasattr(ax, key):
                        getattr(ax, key)(value)
                    elif key.startswith('set_'):
                        getattr(ax, key)(value)
                except Exception as e:
                    logger.warning(f"Failed to apply custom style '{key}': {str(e)}")
        
        fig.tight_layout()
        return fig
    
    def _generate_vehicle_types_report(self, data: Dict[str, Any], config: ReportConfig) -> Figure:
        """
        Generate vehicle types distribution report.
        
        Args:
            data: Dictionary with 'types' and 'counts' lists
            config: Report configuration
        
        Returns:
            Matplotlib Figure object
        """
        # Extract data
        vehicle_types = data.get('types', [])
        counts = data.get('counts', [])
        
        if not vehicle_types or not counts or len(vehicle_types) != len(counts):
            raise ValueError("Invalid data format for vehicle types report")
        
        # Create figure
        fig = plt.figure(figsize=(config.width/config.dpi, config.height/config.dpi), dpi=config.dpi)
        ax = fig.add_subplot(111)
        
        # Plot data
        if config.chart_type == ChartType.BAR:
            ax.bar(vehicle_types, counts, color=config.colors if config.colors else None)
        elif config.chart_type == ChartType.PIE:
            wedges, texts, autotexts = ax.pie(
                counts, 
                labels=vehicle_types, 
                autopct='%1.1f%%',
                colors=config.colors if config.colors else None,
                startangle=90
            )
            # Make labels more readable
            for text in texts:
                text.set_fontsize(10)
            for autotext in autotexts:
                autotext.set_fontsize(8)
                autotext.set_color('white')
            
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        else:
            # Default to bar chart
            ax.bar(vehicle_types, counts, color=config.colors if config.colors else None)
        
        # Configure axis and labels
        if config.chart_type != ChartType.PIE:
            ax.set_xlabel('Vehicle Type')
            ax.set_ylabel('Count')
        
        ax.set_title(config.title)
        
        if config.subtitle:
            fig.suptitle(config.subtitle, fontsize=10)
        
        # Add grid if requested (not for pie chart)
        if config.chart_type != ChartType.PIE:
            ax.grid(config.grid)
        
        # Add annotations for the largest category
        if config.chart_type != ChartType.PIE:
            max_type_index = np.argmax(counts)
            max_type = vehicle_types[max_type_index]
            max_count = counts[max_type_index]
            
            ax.annotate(f"{max_type}: {max_count}",
                      xy=(max_type_index, max_count),
                      xytext=(max_type_index, max_count + max(counts) * 0.1),
                      arrowprops=dict(facecolor='black', shrink=0.05),
                      ha='center')
        
        # Add timestamp if requested
        if config.include_timestamp:
            fig.text(0.99, 0.01, f"Generated: {datetime.datetime.now().strftime(config.date_format)}",
                    ha='right', va='bottom', fontsize=8, color='gray')
        
        # Add logo if requested
        if config.include_logo and config.logo_path and os.path.exists(config.logo_path):
            try:
                logo = plt.imread(config.logo_path)
                ax_logo = fig.add_axes([0.01, 0.01, 0.1, 0.1], frameon=False)
                ax_logo.imshow(logo)
                ax_logo.axis('off')
            except Exception as e:
                logger.warning(f"Failed to add logo to report: {str(e)}")
        
        # Apply custom style if provided
        if config.custom_style:
            for key, value in config.custom_style.items():
                try:
                    if hasattr(ax, key):
                        getattr(ax, key)(value)
                    elif key.startswith('set_'):
                        getattr(ax, key)(value)
                except Exception as e:
                    logger.warning(f"Failed to apply custom style '{key}': {str(e)}")
        
        fig.tight_layout()
        return fig
    
    def _generate_license_plates_report(self, data: Dict[str, Any], config: ReportConfig) -> Figure:
        """
        Generate license plate detection report.
        
        Args:
            data: Dictionary with 'timestamps', 'plates', and optional 'vehicle_ids'
            config: Report configuration
        
        Returns:
            Matplotlib Figure object
        """
        # Extract data
        timestamps = data.get('timestamps', [])
        plates = data.get('plates', [])
        vehicle_ids = data.get('vehicle_ids', [None] * len(plates))
        
        if not timestamps or not plates or len(timestamps) != len(plates):
            raise ValueError("Invalid data format for license plates report")
        
        # Convert timestamps to datetime if they are numeric
        if isinstance(timestamps[0], (int, float)):
            timestamps = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
        
        # Create figure
        fig = plt.figure(figsize=(config.width/config.dpi, config.height/config.dpi), dpi=config.dpi)
        
        # Table is best for this type of data
        if config.chart_type == ChartType.TABLE or True:  # Always use table for this report
            # Hide normal axes
            ax = fig.add_subplot(111)
            ax.axis('off')
            
            # Create table data
            table_data = []
            for ts, plate, vid in zip(timestamps, plates, vehicle_ids):
                ts_str = ts.strftime(config.date_format)
                vid_str = str(vid) if vid is not None else 'N/A'
                table_data.append([ts_str, plate, vid_str])
            
            # Create table
            table = ax.table(
                cellText=table_data,
                colLabels=['Time', 'License Plate', 'Vehicle ID'],
                loc='center',
                cellLoc='center',
                colWidths=[0.4, 0.3, 0.3]
            )
            
            # Style the table
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 1.5)
            
            # Add title
            ax.set_title(config.title, pad=20)
            
            if config.subtitle:
                fig.suptitle(config.subtitle, fontsize=10)
        
        # Add timestamp if requested
        if config.include_timestamp:
            fig.text(0.99, 0.01, f"Generated: {datetime.datetime.now().strftime(config.date_format)}",
                    ha='right', va='bottom', fontsize=8, color='gray')
        
        # Add logo if requested
        if config.include_logo and config.logo_path and os.path.exists(config.logo_path):
            try:
                logo = plt.imread(config.logo_path)
                ax_logo = fig.add_axes([0.01, 0.01, 0.1, 0.1], frameon=False)
                ax_logo.imshow(logo)
                ax_logo.axis('off')
            except Exception as e:
                logger.warning(f"Failed to add logo to report: {str(e)}")
        
        fig.tight_layout()
        return fig
    
    def export_data(
        self,
        data: Any,
        export_format: ExportFormat,
        output_path: Optional[str] = None
    ) -> Union[str, bytes]:
        """
        Export data in various formats.
        
        Args:
            data: Data to export
            export_format: Format for export
            output_path: Optional path to save the export
        
        Returns:
            Path to saved file or bytes data
        """
        if export_format == ExportFormat.CSV:
            result = self._export_csv(data, output_path)
        elif export_format == ExportFormat.JSON:
            result = self._export_json(data, output_path)
        elif export_format == ExportFormat.HTML:
            result = self._export_html(data, output_path)
        else:
            raise ValueError(f"Unsupported export format for data: {export_format}")
        
        return result
    
    def _export_csv(self, data: Any, output_path: Optional[str] = None) -> Union[str, bytes]:
        """
        Export data as CSV.
        
        Args:
            data: Data to export
            output_path: Optional path to save the CSV
        
        Returns:
            Path to saved file or CSV string
        """
        import csv
        import io
        
        # Convert data to list of dictionaries if not already
        if isinstance(data, dict):
            if all(isinstance(v, list) for v in data.values()):
                # Convert columnar data to rows
                keys = list(data.keys())
                rows = []
                for i in range(len(data[keys[0]])):
                    row = {k: data[k][i] for k in keys}
                    rows.append(row)
                data = rows
            else:
                # Single row
                data = [data]
        
        if not data:
            raise ValueError("No data to export")
        
        # Create CSV
        output = io.StringIO()
        fieldnames = data[0].keys()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
        
        csv_data = output.getvalue()
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w', newline='') as f:
                f.write(csv_data)
            return output_path
        
        return csv_data
    
    def _export_json(self, data: Any, output_path: Optional[str] = None) -> Union[str, bytes]:
        """
        Export data as JSON.
        
        Args:
            data: Data to export
            output_path: Optional path to save the JSON
        
        Returns:
            Path to saved file or JSON string
        """
        # Convert data to JSON
        json_data = json.dumps(data, indent=2, default=str)
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(json_data)
            return output_path
        
        return json_data
    
    def _export_html(self, data: Any, output_path: Optional[str] = None) -> Union[str, bytes]:
        """
        Export data as HTML.
        
        Args:
            data: Data to export
            output_path: Optional path to save the HTML
        
        Returns:
            Path to saved file or HTML string
        """
        # Convert data to list of dictionaries if not already
        if isinstance(data, dict):
            if all(isinstance(v, list) for v in data.values()):
                # Convert columnar data to rows
                keys = list(data.keys())
                rows = []
                for i in range(len(data[keys[0]])):
                    row = {k: data[k][i] for k in keys}
                    rows.append(row)
                data = rows
            else:
                # Single row
                data = [data]
        
        if not data:
            raise ValueError("No data to export")
        
        # Create HTML table
        html = ['<html>', '<head>',
                '<style>',
                'table { border-collapse: collapse; width: 100%; }',
                'th, td { text-align: left; padding: 8px; }',
                'tr:nth-child(even) { background-color: #f2f2f2; }',
                'th { background-color: #4CAF50; color: white; }',
                '</style>',
                '</head>', '<body>',
                f'<h1>Traffic Data Export</h1>',
                f'<p>Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>',
                '<table>', '<tr>']
        
        # Add headers
        for key in data[0].keys():
            html.append(f'<th>{key}</th>')
        html.append('</tr>')
        
        # Add rows
        for row in data:
            html.append('<tr>')
            for value in row.values():
                html.append(f'<td>{value}</td>')
            html.append('</tr>')
        
        html.extend(['</table>', '</body>', '</html>'])
        
        html_data = '\n'.join(html)
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html_data)
            return output_path
        
        return html_data
    
    def create_composite_report(
        self,
        reports: List[Tuple[ReportType, Any, Optional[ReportConfig]]],
        title: str,
        output_path: Optional[str] = None,
        export_format: ExportFormat = ExportFormat.PNG,
        layout: Tuple[int, int] = None
    ) -> Union[str, bytes, Figure]:
        """
        Create a composite report with multiple charts.
        
        Args:
            reports: List of (report_type, data, config) tuples
            title: Report title
            output_path: Optional path to save the report
            export_format: Export format
            layout: Optional layout as (rows, cols)
        
        Returns:
            Path to saved file, bytes data, or matplotlib Figure
        """
        # Determine layout if not provided
        if layout is None:
            n = len(reports)
            cols = min(3, n)
            rows = (n + cols - 1) // cols  # Ceiling division
            layout = (rows, cols)
        else:
            rows, cols = layout
        
        # Create figure
        fig = plt.figure(figsize=(cols * 8, rows * 6), dpi=100)
        fig.suptitle(title, fontsize=16)
        
        # Generate each report
        for i, (report_type, data, report_config) in enumerate(reports):
            if i >= rows * cols:
                logger.warning(f"Layout too small for all reports, skipping report {i+1}")
                continue
            
            # Create subplot
            ax = fig.add_subplot(rows, cols, i + 1)
            
            # Generate the report with this axis
            config = report_config or ReportConfig(title=f"{report_type.value}")
            
            # Override certain settings for the subplot
            config.width = int(fig.get_figwidth() * fig.get_dpi() / cols)
            config.height = int(fig.get_figheight() * fig.get_dpi() / rows)
            config.include_timestamp = False
            config.include_logo = False
            
            # Generate the report
            sub_fig = self.generate_report(report_type, data, config)
            
            # Copy the axes to our figure
            if isinstance(sub_fig, Figure):
                # Copy the first axes from the subfigure
                sub_ax = sub_fig.axes[0]
                
                # Copy the plot elements to our subplot
                for line in sub_ax.lines:
                    ax.add_line(line)
                
                for patch in sub_ax.patches:
                    ax.add_patch(patch)
                
                for collection in sub_ax.collections:
                    ax.add_collection(collection)
                
                # Copy axes settings
                ax.set_title(sub_ax.get_title())
                ax.set_xlabel(sub_ax.get_xlabel())
                ax.set_ylabel(sub_ax.get_ylabel())
                ax.set_xlim(sub_ax.get_xlim())
                ax.set_ylim(sub_ax.get_ylim())
                
                # Close the subfigure
                plt.close(sub_fig)
        
        # Adjust layout
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle
        
        # Add timestamp
        fig.text(0.99, 0.01, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ha='right', va='bottom', fontsize=8, color='gray')
        
        # Save or return
        if output_path:
            fig.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            return output_path
        
        return fig
