"""
Observability System for Trading System

Comprehensive observability with OpenTelemetry integration,
structured logging, health endpoints, and performance metrics.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import uuid
import traceback
from contextlib import asynccontextmanager
import psutil
import os

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
# from opentelemetry.instrumentation.kafka import KafkaInstrumentor  # Not available
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.resources import Resource

import structlog
from prometheus_client import Counter, Histogram, Gauge, Summary, generate_latest, CONTENT_TYPE_LATEST


logger = logging.getLogger(__name__)


class TradingTelemetry:
    """
    Comprehensive observability system for the trading platform
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Service identification
        self.service_name = config.get('service_name', 'trading-system')
        self.service_version = config.get('service_version', '1.0.0')
        self.environment = config.get('environment', 'development')
        
        # OpenTelemetry configuration
        self.jaeger_endpoint = config.get('jaeger_endpoint', 'http://localhost:14268/api/traces')
        self.prometheus_port = config.get('prometheus_port', 8000)
        
        # Initialize OpenTelemetry
        self._init_opentelemetry()
        
        # Initialize structured logging
        self._init_structured_logging()
        
        # Initialize Prometheus metrics
        self._init_prometheus_metrics()
        
        # Performance tracking
        self.performance_metrics = {
            'request_count': 0,
            'error_count': 0,
            'avg_response_time_ms': 0.0,
            'p95_response_time_ms': 0.0,
            'p99_response_time_ms': 0.0,
            'throughput_rps': 0.0,
        }
        
        # Health status
        self.is_healthy = True
        self.health_checks = {}
        self.last_health_check = datetime.utcnow()
        
        # Background tasks
        self.background_tasks = []
        
    def _init_opentelemetry(self):
        """Initialize OpenTelemetry tracing and metrics"""
        try:
            # Create resource
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": self.service_version,
                "environment": self.environment,
            })
            
            # Initialize tracer provider
            tracer_provider = TracerProvider(resource=resource)
            
            # Add span processors
            tracer_provider.add_span_processor(
                BatchSpanProcessor(ConsoleSpanExporter())
            )
            
            # Add Jaeger exporter if configured
            if self.jaeger_endpoint:
                jaeger_exporter = JaegerExporter(
                    collector_endpoint=self.jaeger_endpoint
                )
                tracer_provider.add_span_processor(
                    BatchSpanProcessor(jaeger_exporter)
                )
            
            # Set global tracer provider
            trace.set_tracer_provider(tracer_provider)
            self.tracer = trace.get_tracer(self.service_name)
            
            # Initialize meter provider
            metric_reader = PeriodicExportingMetricReader(
                ConsoleMetricExporter(),
                export_interval_millis=60000  # Export every minute
            )
            
            meter_provider = MeterProvider(
                resource=resource,
                metric_readers=[metric_reader]
            )
            
            # Add Prometheus reader if configured
            if self.prometheus_port:
                prometheus_reader = PrometheusMetricReader()
                meter_provider.add_metric_reader(prometheus_reader)
            
            # Set global meter provider
            metrics.set_meter_provider(meter_provider)
            self.meter = metrics.get_meter(self.service_name)
            
            # Create common metrics
            self.request_counter = self.meter.create_counter(
                name="trading_requests_total",
                description="Total number of requests"
            )
            
            self.error_counter = self.meter.create_counter(
                name="trading_errors_total",
                description="Total number of errors"
            )
            
            self.response_time_histogram = self.meter.create_histogram(
                name="trading_response_time_seconds",
                description="Response time in seconds"
            )
            
            self.active_requests_gauge = self.meter.create_up_down_counter(
                name="trading_active_requests",
                description="Number of active requests"
            )
            
            logger.info("✅ OpenTelemetry initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenTelemetry: {e}")
            # Fallback to basic logging
            self.tracer = None
            self.meter = None
    
    def _init_structured_logging(self):
        """Initialize structured logging with structlog"""
        try:
            # Configure structlog
            structlog.configure(
                processors=[
                    structlog.stdlib.filter_by_level,
                    structlog.stdlib.add_logger_name,
                    structlog.stdlib.add_log_level,
                    structlog.stdlib.PositionalArgumentsFormatter(),
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.format_exc_info,
                    structlog.processors.UnicodeDecoder(),
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                wrapper_class=structlog.stdlib.BoundLogger,
                cache_logger_on_first_use=True,
            )
            
            self.structured_logger = structlog.get_logger()
            logger.info("✅ Structured logging initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize structured logging: {e}")
            self.structured_logger = None
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        try:
            # Custom Prometheus metrics
            self.prometheus_metrics = {
                'trading_requests_total': Counter(
                    'trading_requests_total',
                    'Total number of trading requests',
                    ['service', 'endpoint', 'method']
                ),
                'trading_errors_total': Counter(
                    'trading_errors_total',
                    'Total number of trading errors',
                    ['service', 'endpoint', 'error_type']
                ),
                'trading_response_time_seconds': Histogram(
                    'trading_response_time_seconds',
                    'Response time in seconds',
                    ['service', 'endpoint'],
                    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
                ),
                'trading_active_requests': Gauge(
                    'trading_active_requests',
                    'Number of active requests',
                    ['service']
                ),
                'trading_memory_usage_bytes': Gauge(
                    'trading_memory_usage_bytes',
                    'Memory usage in bytes',
                    ['service']
                ),
                'trading_cpu_usage_percent': Gauge(
                    'trading_cpu_usage_percent',
                    'CPU usage percentage',
                    ['service']
                ),
                'trading_feature_store_latency_ms': Histogram(
                    'trading_feature_store_latency_ms',
                    'Feature store read latency in milliseconds',
                    ['service'],
                    buckets=[1, 2, 5, 10, 25, 50, 100]
                ),
                'trading_event_bus_latency_ms': Histogram(
                    'trading_event_bus_latency_ms',
                    'Event bus publish latency in milliseconds',
                    ['service', 'topic'],
                    buckets=[1, 2, 5, 10, 25, 50, 100]
                ),
            }
            
            logger.info("✅ Prometheus metrics initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Prometheus metrics: {e}")
            self.prometheus_metrics = {}
    
    def log_event(self, 
                  event_type: str, 
                  message: str, 
                  trace_id: Optional[str] = None,
                  **kwargs):
        """Log a structured event"""
        try:
            log_data = {
                'event_type': event_type,
                'message': message,
                'timestamp': datetime.utcnow().isoformat(),
                'service': self.service_name,
                'version': self.service_version,
                'environment': self.environment,
            }
            
            if trace_id:
                log_data['trace_id'] = trace_id
            
            # Add additional context
            log_data.update(kwargs)
            
            if self.structured_logger:
                self.structured_logger.info(message, **log_data)
            else:
                logger.info(f"{event_type}: {message} | {json.dumps(log_data)}")
                
        except Exception as e:
            logger.error(f"❌ Failed to log event: {e}")
    
    def log_error(self, 
                  error: Exception, 
                  context: str, 
                  trace_id: Optional[str] = None,
                  **kwargs):
        """Log a structured error"""
        try:
            error_data = {
                'error_type': type(error).__name__,
                'error_message': str(error),
                'error_traceback': traceback.format_exc(),
                'context': context,
                'timestamp': datetime.utcnow().isoformat(),
                'service': self.service_name,
                'version': self.service_version,
                'environment': self.environment,
            }
            
            if trace_id:
                error_data['trace_id'] = trace_id
            
            # Add additional context
            error_data.update(kwargs)
            
            if self.structured_logger:
                self.structured_logger.error(
                    f"Error in {context}: {str(error)}", 
                    **error_data
                )
            else:
                logger.error(f"ERROR in {context}: {str(error)} | {json.dumps(error_data)}")
            
            # Increment error counter
            self._increment_error_counter(context, type(error).__name__)
            
        except Exception as e:
            logger.error(f"❌ Failed to log error: {e}")
    
    @asynccontextmanager
    async def trace_operation(self, operation_name: str, **attributes):
        """Trace an operation with OpenTelemetry"""
        if not self.tracer:
            yield
            return
        
        try:
            with self.tracer.start_as_current_span(operation_name, attributes=attributes) as span:
                start_time = time.time()
                
                # Add span attributes
                span.set_attribute("service.name", self.service_name)
                span.set_attribute("service.version", self.service_version)
                span.set_attribute("environment", self.environment)
                
                # Add custom attributes
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
                
                try:
                    yield span
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise
                finally:
                    # Record metrics
                    duration = time.time() - start_time
                    span.set_attribute("duration_seconds", duration)
                    
                    if self.response_time_histogram:
                        self.response_time_histogram.record(duration)
                    
                    if "trading_response_time_seconds" in self.prometheus_metrics:
                        self.prometheus_metrics["trading_response_time_seconds"].observe(
                            duration, 
                            {"service": self.service_name, "operation": operation_name}
                        )
                        
        except Exception as e:
            logger.error(f"❌ Tracing error: {e}")
            yield
    
    def record_metric(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a custom metric"""
        try:
            # Record in OpenTelemetry
            if self.meter:
                # Create metric if it doesn't exist
                if not hasattr(self, f"_{metric_name}_metric"):
                    setattr(self, f"_{metric_name}_metric", 
                           self.meter.create_histogram(metric_name))
                
                metric = getattr(self, f"_{metric_name}_metric")
                metric.record(value, labels or {})
            
            # Record in Prometheus
            if metric_name in self.prometheus_metrics:
                self.prometheus_metrics[metric_name].observe(value, labels or {})
                
        except Exception as e:
            logger.error(f"❌ Failed to record metric {metric_name}: {e}")
    
    def _increment_error_counter(self, context: str, error_type: str):
        """Increment error counter"""
        try:
            # OpenTelemetry
            if self.error_counter:
                self.error_counter.add(1, {"context": context, "error_type": error_type})
            
            # Prometheus
            if "trading_errors_total" in self.prometheus_metrics:
                self.prometheus_metrics["trading_errors_total"].inc(
                    labels={"service": self.service_name, "context": context, "error_type": error_type}
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to increment error counter: {e}")
    
    def _increment_request_counter(self, endpoint: str, method: str):
        """Increment request counter"""
        try:
            # OpenTelemetry
            if self.request_counter:
                self.request_counter.add(1, {"endpoint": endpoint, "method": method})
            
            # Prometheus
            if "trading_requests_total" in self.prometheus_metrics:
                self.prometheus_metrics["trading_requests_total"].inc(
                    labels={"service": self.service_name, "endpoint": endpoint, "method": method}
                )
                
        except Exception as e:
            logger.error(f"❌ Failed to increment request counter: {e}")
    
    async def start_health_monitoring(self):
        """Start health monitoring background task"""
        async def health_monitor():
            while self.is_healthy:
                try:
                    await self._update_system_metrics()
                    await asyncio.sleep(30)  # Update every 30 seconds
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
                    await asyncio.sleep(30)
        
        self.background_tasks.append(asyncio.create_task(health_monitor()))
    
    async def _update_system_metrics(self):
        """Update system metrics"""
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            if "trading_memory_usage_bytes" in self.prometheus_metrics:
                self.prometheus_metrics["trading_memory_usage_bytes"].set(
                    memory.used, 
                    {"service": self.service_name}
                )
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if "trading_cpu_usage_percent" in self.prometheus_metrics:
                self.prometheus_metrics["trading_cpu_usage_percent"].set(
                    cpu_percent, 
                    {"service": self.service_name}
                )
            
            # Update health check timestamp
            self.last_health_check = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"❌ Failed to update system metrics: {e}")
    
    def register_health_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function"""
        self.health_checks[name] = check_func
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        try:
            health_status = {
                'service': self.service_name,
                'version': self.service_version,
                'environment': self.environment,
                'timestamp': datetime.utcnow().isoformat(),
                'healthy': self.is_healthy,
                'last_health_check': self.last_health_check.isoformat(),
                'checks': {},
                'metrics': self.performance_metrics,
                'system': {
                    'memory_usage_percent': psutil.virtual_memory().percent,
                    'cpu_usage_percent': psutil.cpu_percent(),
                    'disk_usage_percent': psutil.disk_usage('/').percent,
                }
            }
            
            # Run health checks
            for name, check_func in self.health_checks.items():
                try:
                    result = check_func()
                    health_status['checks'][name] = {
                        'status': 'healthy' if result else 'unhealthy',
                        'timestamp': datetime.utcnow().isoformat()
                    }
                except Exception as e:
                    health_status['checks'][name] = {
                        'status': 'error',
                        'error': str(e),
                        'timestamp': datetime.utcnow().isoformat()
                    }
            
            # Update overall health
            all_healthy = all(
                check.get('status') == 'healthy' 
                for check in health_status['checks'].values()
            )
            health_status['healthy'] = all_healthy
            
            return health_status
            
        except Exception as e:
            logger.error(f"❌ Failed to get health status: {e}")
            return {
                'service': self.service_name,
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        try:
            return generate_latest()
        except Exception as e:
            logger.error(f"❌ Failed to generate Prometheus metrics: {e}")
            return ""
    
    async def stop(self):
        """Stop the telemetry system"""
        logger.info("🛑 Stopping telemetry system...")
        
        self.is_healthy = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        logger.info("✅ Telemetry system stopped")


# Global telemetry instance
_telemetry_instance: Optional[TradingTelemetry] = None


def get_telemetry() -> TradingTelemetry:
    """Get the global telemetry instance"""
    global _telemetry_instance
    if _telemetry_instance is None:
        raise RuntimeError("Telemetry not initialized. Call init_telemetry() first.")
    return _telemetry_instance


def init_telemetry(config: Dict[str, Any]) -> TradingTelemetry:
    """Initialize the global telemetry instance"""
    global _telemetry_instance
    _telemetry_instance = TradingTelemetry(config)
    return _telemetry_instance


# Convenience functions
def log_event(event_type: str, message: str, trace_id: Optional[str] = None, **kwargs):
    """Log a structured event using the global telemetry instance"""
    telemetry = get_telemetry()
    telemetry.log_event(event_type, message, trace_id, **kwargs)


def log_error(error: Exception, context: str, trace_id: Optional[str] = None, **kwargs):
    """Log a structured error using the global telemetry instance"""
    telemetry = get_telemetry()
    telemetry.log_error(error, context, trace_id, **kwargs)


@asynccontextmanager
async def trace_operation(operation_name: str, **attributes):
    """Trace an operation using the global telemetry instance"""
    telemetry = get_telemetry()
    async with telemetry.trace_operation(operation_name, **attributes) as span:
        yield span


def record_metric(metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """Record a metric using the global telemetry instance"""
    telemetry = get_telemetry()
    telemetry.record_metric(metric_name, value, labels)
