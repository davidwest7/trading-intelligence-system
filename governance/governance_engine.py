"""
Governance and Audit Engine
Human-in-loop signoff, policy clocks, exception reviews, auto-reporting packs
"""

import yaml
import json
import logging
import asyncio
import schedule
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
import sqlite3
import threading
from queue import Queue
import hashlib
import hmac
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"

class CheckStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    ERROR = "error"

@dataclass
class CheckResult:
    check_id: str
    name: str
    status: CheckStatus
    timestamp: datetime
    details: Dict[str, Any]
    reviewer: Optional[str] = None
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    comments: Optional[str] = None

@dataclass
class ExceptionEvent:
    exception_id: str
    name: str
    severity: Severity
    timestamp: datetime
    description: str
    details: Dict[str, Any]
    escalation_level: str
    status: str = "open"
    resolution_time: Optional[datetime] = None
    resolution_notes: Optional[str] = None

@dataclass
class ApprovalRequest:
    request_id: str
    workflow_type: str
    stage: str
    requester: str
    approver: str
    timestamp: datetime
    details: Dict[str, Any]
    status: ApprovalStatus = ApprovalStatus.PENDING
    comments: Optional[str] = None
    time_limit: timedelta = timedelta(hours=24)

class GovernanceEngine:
    """
    Comprehensive governance and audit engine with human-in-loop signoff,
    policy clocks, exception reviews, and auto-reporting capabilities.
    """
    
    def __init__(self, config_path: str = "governance/checklist.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.db_path = "governance/governance.db"
        self._init_database()
        
        # Initialize components
        self.check_results: Dict[str, CheckResult] = {}
        self.exceptions: Dict[str, ExceptionEvent] = {}
        self.approval_requests: Dict[str, ApprovalRequest] = {}
        self.trading_halted = False
        self.policy_clocks = {}
        
        # Notification settings
        self.notification_config = {
            "email": {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "governance@company.com",
                "password": "your_password_here"
            },
            "slack": {
                "webhook_url": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
            }
        }
        
        # Start background tasks
        self._start_background_tasks()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load governance configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            return {}
    
    def _init_database(self):
        """Initialize SQLite database for governance data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS check_results (
                check_id TEXT PRIMARY KEY,
                name TEXT,
                status TEXT,
                timestamp TEXT,
                details TEXT,
                reviewer TEXT,
                approval_status TEXT,
                comments TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exceptions (
                exception_id TEXT PRIMARY KEY,
                name TEXT,
                severity TEXT,
                timestamp TEXT,
                description TEXT,
                details TEXT,
                escalation_level TEXT,
                status TEXT,
                resolution_time TEXT,
                resolution_notes TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS approval_requests (
                request_id TEXT PRIMARY KEY,
                workflow_type TEXT,
                stage TEXT,
                requester TEXT,
                approver TEXT,
                timestamp TEXT,
                details TEXT,
                status TEXT,
                comments TEXT,
                time_limit TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                log_id TEXT PRIMARY KEY,
                user_id TEXT,
                action TEXT,
                timestamp TEXT,
                details TEXT,
                ip_address TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _start_background_tasks(self):
        """Start background monitoring and reporting tasks."""
        # Start policy clock monitoring
        self._start_policy_clocks()
        
        # Start auto-reporting
        self._start_auto_reporting()
        
        # Start exception monitoring
        self._start_exception_monitoring()
        
        # Start approval workflow monitoring
        self._start_approval_monitoring()
    
    def _start_policy_clocks(self):
        """Start policy clock monitoring."""
        def run_policy_clocks():
            while True:
                try:
                    self._check_policy_clocks()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Error in policy clock monitoring: {e}")
        
        thread = threading.Thread(target=run_policy_clocks, daemon=True)
        thread.start()
    
    def _start_auto_reporting(self):
        """Start auto-reporting system."""
        def run_auto_reporting():
            while True:
                try:
                    self._generate_scheduled_reports()
                    time.sleep(3600)  # Check every hour
                except Exception as e:
                    logger.error(f"Error in auto-reporting: {e}")
        
        thread = threading.Thread(target=run_auto_reporting, daemon=True)
        thread.start()
    
    def _start_exception_monitoring(self):
        """Start exception monitoring."""
        def run_exception_monitoring():
            while True:
                try:
                    self._check_exception_escalations()
                    time.sleep(300)  # Check every 5 minutes
                except Exception as e:
                    logger.error(f"Error in exception monitoring: {e}")
        
        thread = threading.Thread(target=run_exception_monitoring, daemon=True)
        thread.start()
    
    def _start_approval_monitoring(self):
        """Start approval workflow monitoring."""
        def run_approval_monitoring():
            while True:
                try:
                    self._check_approval_timeouts()
                    time.sleep(1800)  # Check every 30 minutes
                except Exception as e:
                    logger.error(f"Error in approval monitoring: {e}")
        
        thread = threading.Thread(target=run_approval_monitoring, daemon=True)
        thread.start()
    
    def run_pre_trading_checks(self) -> Dict[str, CheckResult]:
        """Run all pre-trading checks."""
        results = {}
        
        # Data quality checks
        for check in self.config.get('pre_trading_checks', {}).get('data_quality', []):
            result = self._run_check(check)
            results[check['id']] = result
            
        # Model validation checks
        for check in self.config.get('pre_trading_checks', {}).get('model_validation', []):
            result = self._run_check(check)
            results[check['id']] = result
            
        # Risk management checks
        for check in self.config.get('pre_trading_checks', {}).get('risk_management', []):
            result = self._run_check(check)
            results[check['id']] = result
        
        # Store results
        self._store_check_results(results)
        
        # Check if trading should be halted
        self._evaluate_trading_halt(results)
        
        return results
    
    def run_trading_execution_checks(self) -> Dict[str, CheckResult]:
        """Run trading execution checks."""
        results = {}
        
        # Execution validation checks
        for check in self.config.get('trading_execution', {}).get('execution_validation', []):
            result = self._run_check(check)
            results[check['id']] = result
            
        # Real-time monitoring checks
        for check in self.config.get('trading_execution', {}).get('real_time_monitoring', []):
            result = self._run_check(check)
            results[check['id']] = result
        
        # Store results
        self._store_check_results(results)
        
        return results
    
    def run_post_trading_checks(self) -> Dict[str, CheckResult]:
        """Run post-trading checks."""
        results = {}
        
        # Performance analysis checks
        for check in self.config.get('post_trading_checks', {}).get('performance_analysis', []):
            result = self._run_check(check)
            results[check['id']] = result
            
        # Compliance checks
        for check in self.config.get('post_trading_checks', {}).get('compliance_checks', []):
            result = self._run_check(check)
            results[check['id']] = result
        
        # Store results
        self._store_check_results(results)
        
        return results
    
    def _run_check(self, check_config: Dict[str, Any]) -> CheckResult:
        """Run a single check based on configuration."""
        check_id = check_config['id']
        name = check_config['name']
        
        try:
            # Get check function
            check_function = self._get_check_function(check_config.get('check_function', ''))
            
            if check_function:
                # Run the check
                details = check_function(check_config.get('threshold', {}))
                
                # Determine status
                status = self._determine_check_status(details, check_config.get('threshold', {}))
                
                # Create result
                result = CheckResult(
                    check_id=check_id,
                    name=name,
                    status=status,
                    timestamp=datetime.now(),
                    details=details,
                    reviewer=check_config.get('reviewer'),
                    approval_status=ApprovalStatus.PENDING if check_config.get('approval_required', False) else ApprovalStatus.APPROVED
                )
                
                # If approval required and check failed, create approval request
                if check_config.get('approval_required', False) and status in [CheckStatus.FAILED, CheckStatus.ERROR]:
                    self._create_approval_request(result, check_config)
                
                return result
            else:
                # Mock check result for demonstration
                return CheckResult(
                    check_id=check_id,
                    name=name,
                    status=CheckStatus.PASSED,
                    timestamp=datetime.now(),
                    details={"message": "Check completed successfully"},
                    reviewer=check_config.get('reviewer'),
                    approval_status=ApprovalStatus.APPROVED
                )
                
        except Exception as e:
            logger.error(f"Error running check {check_id}: {e}")
            return CheckResult(
                check_id=check_id,
                name=name,
                status=CheckStatus.ERROR,
                timestamp=datetime.now(),
                details={"error": str(e)},
                reviewer=check_config.get('reviewer'),
                approval_status=ApprovalStatus.PENDING
            )
    
    def _get_check_function(self, function_path: str) -> Optional[Callable]:
        """Get check function by path."""
        # This would typically import and return the actual function
        # For now, return None to use mock results
        return None
    
    def _determine_check_status(self, details: Dict[str, Any], threshold: Dict[str, Any]) -> CheckStatus:
        """Determine check status based on details and thresholds."""
        # Mock logic - in practice, this would compare actual metrics to thresholds
        if 'error' in details:
            return CheckStatus.ERROR
        elif 'warning' in details:
            return CheckStatus.WARNING
        else:
            return CheckStatus.PASSED
    
    def _create_approval_request(self, check_result: CheckResult, check_config: Dict[str, Any]):
        """Create approval request for failed check."""
        request_id = f"AR_{check_result.check_id}_{int(time.time())}"
        
        approval_request = ApprovalRequest(
            request_id=request_id,
            workflow_type="check_approval",
            stage=check_config.get('reviewer', 'unknown'),
            requester="system",
            approver=check_config.get('reviewer', 'unknown'),
            timestamp=datetime.now(),
            details={
                "check_id": check_result.check_id,
                "check_name": check_result.name,
                "status": check_result.status.value,
                "details": check_result.details
            },
            time_limit=timedelta(hours=check_config.get('time_limit', 24))
        )
        
        self.approval_requests[request_id] = approval_request
        self._store_approval_request(approval_request)
        
        # Send notification
        self._send_approval_notification(approval_request)
    
    def _evaluate_trading_halt(self, check_results: Dict[str, CheckResult]):
        """Evaluate if trading should be halted based on check results."""
        critical_failures = [
            result for result in check_results.values()
            if result.status in [CheckStatus.FAILED, CheckStatus.ERROR]
            and result.approval_status == ApprovalStatus.PENDING
        ]
        
        if critical_failures:
            self.halt_trading("Critical check failures detected")
    
    def halt_trading(self, reason: str):
        """Halt trading operations."""
        self.trading_halted = True
        logger.warning(f"Trading halted: {reason}")
        
        # Create exception event
        exception = ExceptionEvent(
            exception_id=f"TH_{int(time.time())}",
            name="Trading Halt",
            severity=Severity.CRITICAL,
            timestamp=datetime.now(),
            description=f"Trading halted: {reason}",
            details={"reason": reason, "check_results": [r.check_id for r in self.check_results.values()]},
            escalation_level="chief_trading_officer"
        )
        
        self.exceptions[exception.exception_id] = exception
        self._store_exception(exception)
        
        # Send notifications
        self._send_critical_notification(exception)
    
    def resume_trading(self, approver: str, reason: str):
        """Resume trading operations."""
        if self.trading_halted:
            self.trading_halted = False
            logger.info(f"Trading resumed by {approver}: {reason}")
            
            # Create audit log entry
            self._log_audit_event(
                user_id=approver,
                action="resume_trading",
                details={"reason": reason}
            )
    
    def _check_policy_clocks(self):
        """Check policy clocks and trigger actions."""
        current_time = datetime.now()
        
        # Check trading halts
        for halt in self.config.get('policy_clocks', {}).get('trading_halts', []):
            if self._should_trigger_halt(halt):
                self.halt_trading(f"Policy clock triggered: {halt['name']}")
        
        # Check review schedules
        for review in self.config.get('policy_clocks', {}).get('review_schedules', []):
            if self._should_trigger_review(review, current_time):
                self._schedule_review(review)
    
    def _should_trigger_halt(self, halt_config: Dict[str, Any]) -> bool:
        """Check if trading halt should be triggered."""
        # Mock implementation - would check actual market conditions
        return False
    
    def _should_trigger_review(self, review_config: Dict[str, Any], current_time: datetime) -> bool:
        """Check if review should be triggered."""
        # Mock implementation - would check actual schedule
        return False
    
    def _schedule_review(self, review_config: Dict[str, Any]):
        """Schedule a review meeting."""
        logger.info(f"Scheduling review: {review_config['name']}")
        # In practice, this would integrate with calendar systems
    
    def _generate_scheduled_reports(self):
        """Generate scheduled reports."""
        current_time = datetime.now()
        
        # Daily reports
        if current_time.hour == 18 and current_time.minute == 0:
            self._generate_daily_reports()
        
        # Weekly reports
        if current_time.weekday() == 4 and current_time.hour == 16 and current_time.minute == 0:  # Friday 4 PM
            self._generate_weekly_reports()
        
        # Monthly reports
        if self._is_last_friday(current_time) and current_time.hour == 14 and current_time.minute == 0:
            self._generate_monthly_reports()
    
    def _generate_daily_reports(self):
        """Generate daily reports."""
        for report_config in self.config.get('auto_reporting', {}).get('daily_reports', []):
            self._generate_report(report_config)
    
    def _generate_weekly_reports(self):
        """Generate weekly reports."""
        for report_config in self.config.get('auto_reporting', {}).get('weekly_reports', []):
            self._generate_report(report_config)
    
    def _generate_monthly_reports(self):
        """Generate monthly reports."""
        for report_config in self.config.get('auto_reporting', {}).get('monthly_reports', []):
            self._generate_report(report_config)
    
    def _generate_report(self, report_config: Dict[str, Any]):
        """Generate a specific report."""
        try:
            report_id = report_config['id']
            report_name = report_config['name']
            
            # Generate report content
            content = self._generate_report_content(report_config)
            
            # Create PDF report
            pdf_path = self._create_pdf_report(report_name, content)
            
            # Send report
            self._send_report(report_config, pdf_path)
            
            logger.info(f"Generated report: {report_name}")
            
        except Exception as e:
            logger.error(f"Error generating report {report_config.get('id', 'unknown')}: {e}")
    
    def _generate_report_content(self, report_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report content based on configuration."""
        content = {}
        
        for content_type in report_config.get('content', []):
            if content_type == 'performance_summary':
                content[content_type] = self._get_performance_summary()
            elif content_type == 'risk_metrics':
                content[content_type] = self._get_risk_metrics()
            elif content_type == 'position_summary':
                content[content_type] = self._get_position_summary()
            elif content_type == 'execution_quality':
                content[content_type] = self._get_execution_quality()
            elif content_type == 'compliance_status':
                content[content_type] = self._get_compliance_status()
            # Add more content types as needed
        
        return content
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary data."""
        # Mock data - would come from actual performance calculations
        return {
            "daily_return": 0.0023,
            "sharpe_ratio": 1.45,
            "information_ratio": 0.87,
            "max_drawdown": -0.0234,
            "volatility": 0.156
        }
    
    def _get_risk_metrics(self) -> Dict[str, Any]:
        """Get risk metrics data."""
        # Mock data - would come from actual risk calculations
        return {
            "var_95": -0.0187,
            "expected_shortfall": -0.0256,
            "beta": 0.89,
            "tracking_error": 0.0234
        }
    
    def _get_position_summary(self) -> Dict[str, Any]:
        """Get position summary data."""
        # Mock data - would come from actual position data
        return {
            "total_positions": 45,
            "long_positions": 28,
            "short_positions": 17,
            "net_exposure": 0.0234,
            "sector_breakdown": {
                "technology": 0.25,
                "healthcare": 0.18,
                "financials": 0.15
            }
        }
    
    def _get_execution_quality(self) -> Dict[str, Any]:
        """Get execution quality data."""
        # Mock data - would come from actual execution data
        return {
            "average_slippage": 0.0008,
            "implementation_shortfall": 0.0012,
            "fill_rate": 0.987,
            "average_commission": 0.0005
        }
    
    def _get_compliance_status(self) -> Dict[str, Any]:
        """Get compliance status data."""
        # Mock data - would come from actual compliance checks
        return {
            "regulatory_compliance": "compliant",
            "policy_violations": 0,
            "trading_restrictions": [],
            "documentation_status": "complete"
        }
    
    def _create_pdf_report(self, report_name: str, content: Dict[str, Any]) -> str:
        """Create PDF report with charts and tables."""
        # Create charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance Metrics', 'Risk Metrics', 'Position Summary', 'Execution Quality'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )
        
        # Performance metrics chart
        if 'performance_summary' in content:
            perf = content['performance_summary']
            fig.add_trace(
                go.Bar(x=['Daily Return', 'Sharpe Ratio', 'Information Ratio'], 
                      y=[perf['daily_return'], perf['sharpe_ratio'], perf['information_ratio']]),
                row=1, col=1
            )
        
        # Risk metrics chart
        if 'risk_metrics' in content:
            risk = content['risk_metrics']
            fig.add_trace(
                go.Scatter(x=['VaR 95%', 'Expected Shortfall', 'Beta', 'Tracking Error'],
                          y=[abs(risk['var_95']), abs(risk['expected_shortfall']), risk['beta'], risk['tracking_error']]),
                row=1, col=2
            )
        
        # Position summary chart
        if 'position_summary' in content:
            pos = content['position_summary']
            fig.add_trace(
                go.Pie(labels=list(pos['sector_breakdown'].keys()),
                      values=list(pos['sector_breakdown'].values())),
                row=2, col=1
            )
        
        # Execution quality chart
        if 'execution_quality' in content:
            exec_qual = content['execution_quality']
            fig.add_trace(
                go.Bar(x=['Slippage', 'Implementation Shortfall', 'Commission'],
                      y=[exec_qual['average_slippage'], exec_qual['implementation_shortfall'], exec_qual['average_commission']]),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text=f"{report_name} - {datetime.now().strftime('%Y-%m-%d')}")
        
        # Save as PDF
        pdf_path = f"governance/reports/{report_name}_{datetime.now().strftime('%Y%m%d')}.pdf"
        Path("governance/reports").mkdir(exist_ok=True)
        fig.write_image(pdf_path)
        
        return pdf_path
    
    def _send_report(self, report_config: Dict[str, Any], pdf_path: str):
        """Send report to recipients."""
        recipients = report_config.get('recipients', [])
        formats = report_config.get('format', ['email'])
        
        if 'email' in formats:
            self._send_email_report(recipients, report_config['name'], pdf_path)
        
        if 'slack' in formats:
            self._send_slack_report(recipients, report_config['name'], pdf_path)
    
    def _send_email_report(self, recipients: List[str], report_name: str, pdf_path: str):
        """Send report via email."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.notification_config['email']['username']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"Daily Report: {report_name}"
            
            body = f"""
            Please find attached the {report_name} for {datetime.now().strftime('%Y-%m-%d')}.
            
            This report was automatically generated by the Trading Intelligence System Governance Engine.
            
            Best regards,
            Governance System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Attach PDF
            with open(pdf_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
            
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {Path(pdf_path).name}'
            )
            msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(self.notification_config['email']['smtp_server'], 
                                self.notification_config['email']['smtp_port'])
            server.starttls()
            server.login(self.notification_config['email']['username'],
                        self.notification_config['email']['password'])
            text = msg.as_string()
            server.sendmail(self.notification_config['email']['username'], recipients, text)
            server.quit()
            
            logger.info(f"Email report sent to {recipients}")
            
        except Exception as e:
            logger.error(f"Error sending email report: {e}")
    
    def _send_slack_report(self, recipients: List[str], report_name: str, pdf_path: str):
        """Send report via Slack."""
        try:
            message = {
                "text": f"📊 {report_name} Report",
                "attachments": [
                    {
                        "title": f"{report_name} - {datetime.now().strftime('%Y-%m-%d')}",
                        "text": f"Automatically generated report from Trading Intelligence System",
                        "color": "#36a64f"
                    }
                ]
            }
            
            response = requests.post(
                self.notification_config['slack']['webhook_url'],
                json=message
            )
            
            if response.status_code == 200:
                logger.info(f"Slack report sent")
            else:
                logger.error(f"Error sending Slack report: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending Slack report: {e}")
    
    def _check_exception_escalations(self):
        """Check for exception escalations."""
        current_time = datetime.now()
        
        for exception in self.exceptions.values():
            if exception.status == "open":
                escalation_time = self._get_escalation_time(exception.escalation_level)
                
                if current_time - exception.timestamp > escalation_time:
                    self._escalate_exception(exception)
    
    def _get_escalation_time(self, escalation_level: str) -> timedelta:
        """Get escalation time for a given level."""
        escalation_times = {
            "data_engineer": timedelta(minutes=30),
            "quantitative_researcher": timedelta(minutes=15),
            "risk_manager": timedelta(minutes=5),
            "trader": timedelta(minutes=10),
            "system_administrator": timedelta(minutes=5)
        }
        
        return escalation_times.get(escalation_level, timedelta(hours=1))
    
    def _escalate_exception(self, exception: ExceptionEvent):
        """Escalate an exception."""
        logger.warning(f"Escalating exception {exception.exception_id} to {exception.escalation_level}")
        
        # Send escalation notification
        self._send_escalation_notification(exception)
        
        # Update exception status
        exception.status = "escalated"
        self._store_exception(exception)
    
    def _check_approval_timeouts(self):
        """Check for approval request timeouts."""
        current_time = datetime.now()
        
        for request in self.approval_requests.values():
            if request.status == ApprovalStatus.PENDING:
                if current_time - request.timestamp > request.time_limit:
                    self._handle_approval_timeout(request)
    
    def _handle_approval_timeout(self, request: ApprovalRequest):
        """Handle approval request timeout."""
        logger.warning(f"Approval request {request.request_id} timed out")
        
        # Auto-escalate or take default action
        request.status = ApprovalStatus.ESCALATED
        self._store_approval_request(request)
        
        # Send timeout notification
        self._send_timeout_notification(request)
    
    def _send_approval_notification(self, request: ApprovalRequest):
        """Send approval request notification."""
        message = f"""
        Approval Request: {request.workflow_type} - {request.stage}
        
        Requester: {request.requester}
        Approver: {request.approver}
        Time Limit: {request.time_limit}
        
        Details: {json.dumps(request.details, indent=2)}
        
        Please review and approve/reject this request.
        """
        
        self._send_notification(request.approver, "Approval Request", message)
    
    def _send_critical_notification(self, exception: ExceptionEvent):
        """Send critical notification."""
        message = f"""
        CRITICAL ALERT: {exception.name}
        
        Severity: {exception.severity.value}
        Description: {exception.description}
        Escalation Level: {exception.escalation_level}
        
        Details: {json.dumps(exception.details, indent=2)}
        
        Immediate action required.
        """
        
        self._send_notification(exception.escalation_level, "Critical Alert", message)
    
    def _send_escalation_notification(self, exception: ExceptionEvent):
        """Send escalation notification."""
        message = f"""
        ESCALATION: {exception.name}
        
        Exception {exception.exception_id} has been escalated to {exception.escalation_level}.
        
        Description: {exception.description}
        Severity: {exception.severity.value}
        
        Immediate attention required.
        """
        
        self._send_notification(exception.escalation_level, "Exception Escalation", message)
    
    def _send_timeout_notification(self, request: ApprovalRequest):
        """Send timeout notification."""
        message = f"""
        APPROVAL TIMEOUT: {request.workflow_type} - {request.stage}
        
        Request {request.request_id} has timed out.
        Requester: {request.requester}
        Approver: {request.approver}
        
        Action required.
        """
        
        self._send_notification(request.approver, "Approval Timeout", message)
    
    def _send_notification(self, recipient: str, subject: str, message: str):
        """Send notification to recipient."""
        # Send email notification
        try:
            msg = MIMEText(message)
            msg['Subject'] = subject
            msg['From'] = self.notification_config['email']['username']
            msg['To'] = f"{recipient}@company.com"
            
            server = smtplib.SMTP(self.notification_config['email']['smtp_server'], 
                                self.notification_config['email']['smtp_port'])
            server.starttls()
            server.login(self.notification_config['email']['username'],
                        self.notification_config['email']['password'])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.error(f"Error sending notification to {recipient}: {e}")
    
    def _store_check_results(self, results: Dict[str, CheckResult]):
        """Store check results in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for result in results.values():
            cursor.execute('''
                INSERT OR REPLACE INTO check_results 
                (check_id, name, status, timestamp, details, reviewer, approval_status, comments)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.check_id,
                result.name,
                result.status.value,
                result.timestamp.isoformat(),
                json.dumps(result.details),
                result.reviewer,
                result.approval_status.value,
                result.comments
            ))
        
        conn.commit()
        conn.close()
    
    def _store_exception(self, exception: ExceptionEvent):
        """Store exception in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO exceptions 
            (exception_id, name, severity, timestamp, description, details, escalation_level, status, resolution_time, resolution_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            exception.exception_id,
            exception.name,
            exception.severity.value,
            exception.timestamp.isoformat(),
            exception.description,
            json.dumps(exception.details),
            exception.escalation_level,
            exception.status,
            exception.resolution_time.isoformat() if exception.resolution_time else None,
            exception.resolution_notes
        ))
        
        conn.commit()
        conn.close()
    
    def _store_approval_request(self, request: ApprovalRequest):
        """Store approval request in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO approval_requests 
            (request_id, workflow_type, stage, requester, approver, timestamp, details, status, comments, time_limit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            request.request_id,
            request.workflow_type,
            request.stage,
            request.requester,
            request.approver,
            request.timestamp.isoformat(),
            json.dumps(request.details),
            request.status.value,
            request.comments,
            request.time_limit.total_seconds()
        ))
        
        conn.commit()
        conn.close()
    
    def _log_audit_event(self, user_id: str, action: str, details: Dict[str, Any], ip_address: str = "127.0.0.1"):
        """Log audit event."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        log_id = hashlib.md5(f"{user_id}_{action}_{time.time()}".encode()).hexdigest()
        
        cursor.execute('''
            INSERT INTO audit_log 
            (log_id, user_id, action, timestamp, details, ip_address)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            log_id,
            user_id,
            action,
            datetime.now().isoformat(),
            json.dumps(details),
            ip_address
        ))
        
        conn.commit()
        conn.close()
    
    def _is_last_friday(self, current_time: datetime) -> bool:
        """Check if current time is the last Friday of the month."""
        # Get the last day of the month
        last_day = (current_time.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        
        # Check if it's Friday and the last Friday of the month
        return (current_time.weekday() == 4 and  # Friday
                current_time.day >= last_day.day - 6)  # Within last week
    
    def get_governance_summary(self) -> Dict[str, Any]:
        """Get governance system summary."""
        return {
            "trading_halted": self.trading_halted,
            "total_checks": len(self.check_results),
            "failed_checks": len([r for r in self.check_results.values() if r.status in [CheckStatus.FAILED, CheckStatus.ERROR]]),
            "pending_approvals": len([r for r in self.approval_requests.values() if r.status == ApprovalStatus.PENDING]),
            "open_exceptions": len([e for e in self.exceptions.values() if e.status == "open"]),
            "last_updated": datetime.now().isoformat()
        }
    
    def approve_check(self, check_id: str, approver: str, approved: bool, comments: str = ""):
        """Approve or reject a check."""
        if check_id in self.check_results:
            result = self.check_results[check_id]
            result.approval_status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
            result.comments = comments
            result.reviewer = approver
            
            # Log audit event
            self._log_audit_event(
                user_id=approver,
                action="check_approval",
                details={
                    "check_id": check_id,
                    "approved": approved,
                    "comments": comments
                }
            )
            
            # Store updated result
            self._store_check_results({check_id: result})
            
            logger.info(f"Check {check_id} {'approved' if approved else 'rejected'} by {approver}")
    
    def resolve_exception(self, exception_id: str, resolver: str, resolution_notes: str):
        """Resolve an exception."""
        if exception_id in self.exceptions:
            exception = self.exceptions[exception_id]
            exception.status = "resolved"
            exception.resolution_time = datetime.now()
            exception.resolution_notes = resolution_notes
            
            # Log audit event
            self._log_audit_event(
                user_id=resolver,
                action="exception_resolution",
                details={
                    "exception_id": exception_id,
                    "resolution_notes": resolution_notes
                }
            )
            
            # Store updated exception
            self._store_exception(exception)
            
            logger.info(f"Exception {exception_id} resolved by {resolver}")
    
    def approve_request(self, request_id: str, approver: str, approved: bool, comments: str = ""):
        """Approve or reject an approval request."""
        if request_id in self.approval_requests:
            request = self.approval_requests[request_id]
            request.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
            request.comments = comments
            
            # Log audit event
            self._log_audit_event(
                user_id=approver,
                action="request_approval",
                details={
                    "request_id": request_id,
                    "approved": approved,
                    "comments": comments
                }
            )
            
            # Store updated request
            self._store_approval_request(request)
            
            logger.info(f"Request {request_id} {'approved' if approved else 'rejected'} by {approver}")


# Example usage
if __name__ == "__main__":
    # Initialize governance engine
    governance = GovernanceEngine()
    
    # Run pre-trading checks
    print("Running pre-trading checks...")
    results = governance.run_pre_trading_checks()
    
    # Print results
    for check_id, result in results.items():
        print(f"{check_id}: {result.status.value} - {result.name}")
    
    # Get governance summary
    summary = governance.get_governance_summary()
    print(f"\nGovernance Summary: {summary}")
    
    # Check if trading is halted
    if governance.trading_halted:
        print("⚠️  Trading is currently halted")
    else:
        print("✅ Trading is active")
