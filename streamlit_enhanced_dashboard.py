"""
Enhanced Streamlit Dashboard with Multiple Screens
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import asyncio
import threading
import time
from typing import Dict, List, Any
import json

# Add current directory to path
import sys
sys.path.append('.')

from common.opportunity_store import OpportunityStore
from common.unified_opportunity_scorer_enhanced import EnhancedUnifiedOpportunityScorer
from alternative_data.real_time_data_integration import RealTimeAlternativeData
from ml_models.advanced_ml_models import AdvancedMLPredictor, AdvancedSentimentAnalyzer
from execution_algorithms.advanced_execution import AdvancedExecutionEngine
from hft.high_frequency_trading import HighFrequencyTradingEngine
from risk_management.advanced_risk_manager import AdvancedRiskManager
from common.data_adapters.multi_asset_adapter import MultiAssetDataAdapter


class EnhancedStreamlitDashboard:
    """Enhanced Streamlit Dashboard with multiple screens"""
    
    def __init__(self):
        self.opportunity_store = OpportunityStore()
        self.scorer = EnhancedUnifiedOpportunityScorer()
        self.alternative_data = None
        self.execution_engine = None
        self.hft_engine = None
        self.risk_manager = None
        self.multi_asset_adapter = None
        
        # Initialize session state
        if 'current_screen' not in st.session_state:
            st.session_state.current_screen = 'Top Opportunities'
        if 'opportunities_data' not in st.session_state:
            st.session_state.opportunities_data = []
        if 'open_positions' not in st.session_state:
            st.session_state.open_positions = []
        if 'pending_positions' not in st.session_state:
            st.session_state.pending_positions = []
        if 'market_sentiment' not in st.session_state:
            st.session_state.market_sentiment = {}
        
    async def initialize_components(self):
        """Initialize all backend components"""
        try:
            # Initialize alternative data
            self.alternative_data = RealTimeAlternativeData()
            await self.alternative_data.initialize()
            
            # Initialize execution engine
            self.execution_engine = AdvancedExecutionEngine()
            await self.execution_engine.initialize()
            
            # Initialize HFT engine
            self.hft_engine = HighFrequencyTradingEngine()
            await self.hft_engine.initialize()
            
            # Initialize risk manager
            self.risk_manager = AdvancedRiskManager()
            
            # Initialize multi-asset adapter
            config = {
                'alpha_vantage_key': 'demo',
                'binance_api_key': 'demo',
                'fxcm_api_key': 'demo'
            }
            self.multi_asset_adapter = MultiAssetDataAdapter(config)
            await self.multi_asset_adapter.connect()
            
            return True
        except Exception as e:
            st.error(f"Error initializing components: {e}")
            return False
    
    def create_navigation_menu(self):
        """Create the top navigation menu"""
        st.markdown("""
        <style>
        .nav-menu {
            display: flex;
            justify-content: space-between;
            background-color: #1f1f1f;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .nav-button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin: 0 5px;
        }
        .nav-button:hover {
            background-color: #45a049;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Trading Analytics Group
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("📊 Top Opportunities", key="nav_top_opps"):
                st.session_state.current_screen = 'Top Opportunities'
        
        with col2:
            if st.button("📈 Open Positions", key="nav_open_pos"):
                st.session_state.current_screen = 'Open Positions'
        
        with col3:
            if st.button("⏳ Pending Positions", key="nav_pending_pos"):
                st.session_state.current_screen = 'Pending Positions'
        
        with col4:
            if st.button("📋 Account Strategy", key="nav_account_strat"):
                st.session_state.current_screen = 'Account Strategy'
        
        with col5:
            if st.button("📊 Trading Analytics", key="nav_trading_analytics"):
                st.session_state.current_screen = 'Trading Analytics'
        
        # Market Analytics Group
        col6, col7, col8, col9, col10 = st.columns(5)
        
        with col6:
            if st.button("🌍 Market Sentiment", key="nav_market_sentiment"):
                st.session_state.current_screen = 'Market Sentiment'
        
        with col7:
            if st.button("🏭 Industry Analytics", key="nav_industry_analytics"):
                st.session_state.current_screen = 'Industry Analytics'
        
        with col8:
            if st.button("📈 Top Industries", key="nav_top_industries"):
                st.session_state.current_screen = 'Top Industries'
        
        with col9:
            if st.button("📉 Worst Industries", key="nav_worst_industries"):
                st.session_state.current_screen = 'Worst Industries'
        
        with col10:
            if st.button("📊 Real-time Fundamentals", key="nav_realtime_fundamentals"):
                st.session_state.current_screen = 'Real-time Fundamentals'
        
        # Technical & ML Group
        col11, col12 = st.columns(2)
        
        with col11:
            if st.button("🔧 Technical Analytics", key="nav_technical_analytics"):
                st.session_state.current_screen = 'Technical Analytics'
        
        with col12:
            if st.button("🤖 Model Learning", key="nav_model_learning"):
                st.session_state.current_screen = 'Model Learning'
    
    def generate_mock_opportunities(self):
        """Generate mock opportunities data"""
        opportunities = []
        
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'BTC', 'ETH', 'EUR/USD', 'GOLD', 'OIL']
        order_types = ['Market', 'Limit', 'Stop', 'Stop Limit']
        reasoning_types = ['Technical Breakout', 'Fundamental Value', 'Sentiment Analysis', 'ML Prediction', 'Arbitrage']
        
        for i in range(20):
            symbol = np.random.choice(symbols)
            current_price = np.random.uniform(50, 500)
            target_price = current_price * np.random.uniform(0.95, 1.15)
            stop_loss = current_price * np.random.uniform(0.85, 0.98)
            
            opportunity = {
                'ticker': symbol,
                'current_price': round(current_price, 2),
                'target_price': round(target_price, 2),
                'stop_loss': round(stop_loss, 2),
                'market_order_type': np.random.choice(order_types),
                'risk': round(np.random.uniform(0.01, 0.05), 3),
                'position_size': round(np.random.uniform(1000, 10000), 0),
                'reasoning': np.random.choice(reasoning_types),
                'confidence': round(np.random.uniform(0.6, 0.95), 2),
                'expected_return': round((target_price - current_price) / current_price * 100, 2),
                'timestamp': datetime.now() - timedelta(hours=np.random.randint(0, 24)),
                'status': np.random.choice(['Open', 'Pending', 'Closed'])
            }
            opportunities.append(opportunity)
        
        return opportunities
    
    def create_opportunities_table(self, opportunities):
        """Create the opportunities table"""
        if not opportunities:
            st.warning("No opportunities available")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(opportunities)
        
        # Create the table with custom styling
        st.markdown("### 📊 Top Opportunity Positions")
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        with col1:
            status_filter = st.selectbox("Status Filter", ['All'] + list(df['status'].unique()))
        with col2:
            confidence_filter = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.1)
        with col3:
            return_filter = st.slider("Min Expected Return (%)", -20.0, 50.0, -20.0, 1.0)
        
        # Apply filters
        filtered_df = df.copy()
        if status_filter != 'All':
            filtered_df = filtered_df[filtered_df['status'] == status_filter]
        filtered_df = filtered_df[filtered_df['confidence'] >= confidence_filter]
        filtered_df = filtered_df[filtered_df['expected_return'] >= return_filter]
        
        # Display table
        st.dataframe(
            filtered_df[[
                'ticker', 'current_price', 'target_price', 'stop_loss', 
                'market_order_type', 'risk', 'position_size', 'reasoning',
                'confidence', 'expected_return', 'status', 'timestamp'
            ]].round(2),
            use_container_width=True,
            hide_index=True
        )
        
        # Add reasoning visualization button
        if st.button("🔍 Show Decision Reasoning Visualization"):
            self.show_reasoning_visualization(filtered_df.iloc[0])
    
    def show_reasoning_visualization(self, opportunity):
        """Show reasoning visualization (mind map style)"""
        st.markdown("### 🧠 Decision Reasoning Visualization")
        
        # Create a simple mind map visualization
        fig = go.Figure()
        
        # Central node (opportunity)
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers+text',
            marker=dict(size=30, color='red'),
            text=[opportunity['ticker']],
            textposition="middle center",
            name='Opportunity'
        ))
        
        # Reasoning nodes
        reasoning_factors = {
            'Technical Analysis': [1, 1],
            'Fundamental Analysis': [-1, 1],
            'Sentiment Analysis': [1, -1],
            'ML Prediction': [-1, -1],
            'Market Conditions': [0, 2]
        }
        
        colors = ['blue', 'green', 'orange', 'purple', 'brown']
        
        for i, (factor, pos) in enumerate(reasoning_factors.items()):
            # Add factor node
            fig.add_trace(go.Scatter(
                x=[pos[0]], y=[pos[1]],
                mode='markers+text',
                marker=dict(size=20, color=colors[i]),
                text=[factor],
                textposition="middle center",
                name=factor
            ))
            
            # Add connection line
            fig.add_trace(go.Scatter(
                x=[0, pos[0]], y=[0, pos[1]],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False
            ))
        
        fig.update_layout(
            title=f"Decision Reasoning for {opportunity['ticker']}",
            xaxis=dict(range=[-2, 2], showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(range=[-2, 2], showgrid=False, zeroline=False, showticklabels=False),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show reasoning details
        st.markdown("#### 📋 Reasoning Details")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Confidence Score", f"{opportunity['confidence']:.2%}")
            st.metric("Expected Return", f"{opportunity['expected_return']:.2f}%")
            st.metric("Risk Level", f"{opportunity['risk']:.3f}")
        
        with col2:
            st.metric("Position Size", f"${opportunity['position_size']:,.0f}")
            st.metric("Current Price", f"${opportunity['current_price']:.2f}")
            st.metric("Target Price", f"${opportunity['target_price']:.2f}")
    
    def create_summary_statistics(self, opportunities):
        """Create summary statistics at the bottom"""
        st.markdown("---")
        st.markdown("### 📈 Summary Statistics")
        
        if not opportunities:
            return
        
        df = pd.DataFrame(opportunities)
        
        # Calculate statistics
        daily_opportunities = len(df[df['timestamp'] >= datetime.now() - timedelta(days=1)])
        overall_predicted_returns = df['expected_return'].mean()
        current_open_positions = len(df[df['status'] == 'Open'])
        pending_positions = len(df[df['status'] == 'Pending'])
        total_confidence = df['confidence'].mean()
        total_risk = df['risk'].mean()
        
        # Display metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric("Daily Opportunities", daily_opportunities)
        
        with col2:
            st.metric("Overall Predicted Returns", f"{overall_predicted_returns:.2f}%")
        
        with col3:
            st.metric("Current Open Positions", current_open_positions)
        
        with col4:
            st.metric("Pending Positions", pending_positions)
        
        with col5:
            st.metric("Average Confidence", f"{total_confidence:.2%}")
        
        with col6:
            st.metric("Average Risk", f"{total_risk:.3f}")
        
        # Add charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Opportunities by status
            status_counts = df['status'].value_counts()
            fig1 = px.pie(values=status_counts.values, names=status_counts.index, title="Opportunities by Status")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Expected returns distribution
            fig2 = px.histogram(df, x='expected_return', title="Expected Returns Distribution")
            st.plotly_chart(fig2, use_container_width=True)
    
    def show_market_sentiment_screen(self):
        """Show market sentiment screen"""
        st.markdown("## 🌍 Market Sentiment Analysis")
        
        # Mock sentiment data
        countries = ['US', 'UK', 'EU', 'China', 'Japan', 'South Korea']
        industries = ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer', 'Industrial']
        
        # Country sentiment
        st.markdown("### �� Country Sentiment")
        country_sentiment = {
            'US': 0.75, 'UK': 0.65, 'EU': 0.60, 
            'China': 0.45, 'Japan': 0.70, 'South Korea': 0.80
        }
        
        fig = px.bar(
            x=list(country_sentiment.keys()),
            y=list(country_sentiment.values()),
            title="Market Sentiment by Country",
            labels={'x': 'Country', 'y': 'Sentiment Score'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Industry sentiment
        st.markdown("### 🏭 Industry Sentiment")
        industry_sentiment = {
            'Technology': 0.85, 'Healthcare': 0.70, 'Finance': 0.60,
            'Energy': 0.45, 'Consumer': 0.75, 'Industrial': 0.65
        }
        
        fig2 = px.bar(
            x=list(industry_sentiment.keys()),
            y=list(industry_sentiment.values()),
            title="Market Sentiment by Industry",
            labels={'x': 'Industry', 'y': 'Sentiment Score'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    def show_technical_analytics_screen(self):
        """Show technical analytics screen"""
        st.markdown("## 🔧 Technical Analytics Insights")
        
        # Mock technical indicators
        indicators = ['RSI', 'MACD', 'Bollinger Bands', 'Moving Averages', 'Volume', 'Stochastic']
        values = [65, 0.5, 0.7, 0.8, 0.6, 0.75]
        
        # Technical indicators chart
        fig = px.bar(
            x=indicators,
            y=values,
            title="Technical Indicators Analysis",
            labels={'x': 'Indicator', 'y': 'Signal Strength'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Pattern recognition
        st.markdown("### 📈 Pattern Recognition")
        patterns = ['Double Bottom', 'Head & Shoulders', 'Triangle', 'Flag', 'Cup & Handle']
        confidence = [0.85, 0.70, 0.90, 0.75, 0.80]
        
        pattern_df = pd.DataFrame({
            'Pattern': patterns,
            'Confidence': confidence
        })
        
        st.dataframe(pattern_df, use_container_width=True)
    
    def show_model_learning_screen(self):
        """Show model learning/improvement screen"""
        st.markdown("## 🤖 Model Learning & Improvement")
        
        # Model performance metrics
        st.markdown("### 📊 Model Performance")
        
        models = ['LSTM', 'Transformer', 'Random Forest', 'Gradient Boosting', 'Ensemble']
        accuracy = [0.78, 0.82, 0.75, 0.80, 0.85]
        precision = [0.76, 0.81, 0.74, 0.79, 0.84]
        recall = [0.77, 0.80, 0.73, 0.78, 0.83]
        
        model_df = pd.DataFrame({
            'Model': models,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall
        })
        
        st.dataframe(model_df, use_container_width=True)
        
        # Learning progress
        st.markdown("### 📈 Learning Progress")
        
        epochs = list(range(1, 21))
        training_loss = [0.5, 0.45, 0.42, 0.38, 0.35, 0.32, 0.30, 0.28, 0.26, 0.25,
                        0.24, 0.23, 0.22, 0.21, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15]
        validation_loss = [0.52, 0.48, 0.45, 0.42, 0.40, 0.38, 0.37, 0.36, 0.35, 0.34,
                          0.33, 0.32, 0.31, 0.30, 0.29, 0.28, 0.27, 0.26, 0.25, 0.24]
        
        fig = px.line(
            x=epochs,
            y=[training_loss, validation_loss],
            title="Training Progress",
            labels={'x': 'Epoch', 'y': 'Loss'},
            
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def run_dashboard(self):
        """Run the enhanced dashboard"""
        st.set_page_config(
            page_title="Trading Intelligence System",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🚀 Trading Intelligence System")
        st.markdown("### World-Class Multi-Asset Trading Platform")
        
        # Create navigation menu
        self.create_navigation_menu()
        
        # Initialize components (simplified for demo)
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            # Note: In production, you would await self.initialize_components()
        
        # Generate mock data
        if not st.session_state.opportunities_data:
            st.session_state.opportunities_data = self.generate_mock_opportunities()
        
        # Display current screen
        current_screen = st.session_state.current_screen
        
        if current_screen == 'Top Opportunities':
            self.create_opportunities_table(st.session_state.opportunities_data)
            self.create_summary_statistics(st.session_state.opportunities_data)
        
        elif current_screen == 'Market Sentiment':
            self.show_market_sentiment_screen()
        
        elif current_screen == 'Technical Analytics':
            self.show_technical_analytics_screen()
        
        elif current_screen == 'Model Learning':
            self.show_model_learning_screen()
        
        else:
            st.markdown(f"## {current_screen}")
            st.info(f"Screen '{current_screen}' is under development. This will show {current_screen.lower()} data and analytics.")


def main():
    """Main function to run the dashboard"""
    dashboard = EnhancedStreamlitDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()
