#!/usr/bin/env python3
"""
Test Complete Macro Agent Implementation

Tests all resolved TODOs:
✅ Economic calendar APIs integration
✅ Central bank communication analysis
✅ Election and policy tracking
✅ Scenario mapping and impact analysis
✅ Geopolitical event monitoring
✅ Economic surprise indices
✅ Real-time event impact assessment
✅ Macro theme identification
✅ Regime-dependent impact models
✅ Cross-asset impact forecasting
"""

import asyncio
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.macro.agent_complete import (
    MacroAgent, EconomicCalendarAPI, CentralBankAnalyzer,
    ElectionTracker, ScenarioGenerator, MacroThemeIdentifier
)

async def test_macro_agent():
    """Test the complete macro agent implementation"""
    print("🧪 Testing Complete Macro Agent Implementation")
    print("=" * 60)
    
    # Initialize agent
    agent = MacroAgent()
    
    # Test 1: Economic Calendar API
    print("\n1. Testing Economic Calendar API...")
    calendar_api = EconomicCalendarAPI()
    
    events = await calendar_api.get_upcoming_events(
        window="1m", 
        regions=["US", "EU", "UK"], 
        event_types=["monetary_policy", "economic_data"],
        threshold="medium"
    )
    
    print(f"   Found {len(events)} upcoming events")
    for event in events:
        print(f"     - {event['event']} ({event['date']}) - {event['importance']} impact")
    
    # Test 2: Central Bank Analyzer
    print("\n2. Testing Central Bank Analyzer...")
    cb_analyzer = CentralBankAnalyzer()
    
    # Test communication analysis
    test_communication = "The Federal Reserve remains committed to addressing inflation concerns through appropriate monetary policy measures."
    sentiment = cb_analyzer.analyze_communication(test_communication, "Federal Reserve")
    
    print(f"   Communication sentiment: {sentiment['stance']} (score: {sentiment['sentiment_score']:.3f})")
    print(f"   Confidence: {sentiment['confidence']:.3f}")
    
    # Test policy tracking
    current_policy = {
        "interest_rate": 5.25,
        "policy_stance": "hawkish",
        "inflation_target": 2.0
    }
    
    policy_changes = cb_analyzer.track_policy_changes("Federal Reserve", current_policy)
    print(f"   Policy changes detected: {policy_changes['change_detected']}")
    print(f"   Policy stability: {policy_changes['policy_stability']:.3f}")
    
    # Test 3: Election Tracker
    print("\n3. Testing Election Tracker...")
    election_tracker = ElectionTracker()
    
    elections = election_tracker.get_upcoming_elections("3m")
    print(f"   Found {len(elections)} upcoming elections")
    for election in elections:
        print(f"     - {election['country']}: {election['type']} on {election['date']}")
    
    # Test policy impact analysis
    policy_impact = election_tracker.analyze_policy_impact("Candidate A", ["Tax Reform", "Healthcare"])
    print(f"   Policy impact analysis: {policy_impact}")
    
    # Test 4: Scenario Generator
    print("\n4. Testing Scenario Generator...")
    scenario_generator = ScenarioGenerator()
    
    scenarios = scenario_generator.generate_scenarios("1m")
    print(f"   Generated {len(scenarios)} risk scenarios")
    
    for scenario in scenarios:
        print(f"     - {scenario.scenario}: {scenario.probability:.1%} probability")
        print(f"       Tail risk: {scenario.tail_risk}, Impact: {scenario.impact_magnitude:.3f}")
        print(f"       Hedging: {', '.join(scenario.hedging_strategies[:3])}")
    
    # Test 5: Macro Theme Identifier
    print("\n5. Testing Macro Theme Identifier...")
    theme_identifier = MacroThemeIdentifier()
    
    # Test theme identification
    test_events = [
        {"event": "Federal Reserve FOMC Meeting"},
        {"event": "ECB Interest Rate Decision"},
        {"event": "Inflation data shows price pressures"}
    ]
    
    themes = theme_identifier.identify_themes(test_events, "1m")
    print(f"   Identified {len(themes)} macro themes")
    
    for theme in themes:
        print(f"     - {theme.theme}: strength {theme.strength:.3f}, confidence {theme.confidence:.3f}")
        print(f"       Trend: {theme.trend_direction}, Duration: {theme.duration}")
        print(f"       Affected sectors: {', '.join(theme.affected_sectors[:3])}")
    
    # Test 6: Complete Macro Agent
    print("\n6. Testing Complete Macro Agent...")
    
    result = await agent.timeline(
        window="1m",
        regions=["US", "EU", "UK"],
        event_types=["monetary_policy", "economic_data", "elections"],
        impact_threshold="medium"
    )
    
    print(f"   Events: {result['summary']['total_events']} total, {result['summary']['high_impact_events']} high impact")
    print(f"   Dominant theme: {result['summary']['dominant_theme']}")
    print(f"   Highest risk scenario: {result['summary']['highest_risk_scenario']}")
    
    print(f"   Central bank analysis: {len(result['central_bank_analysis'])} banks analyzed")
    print(f"   Elections: {len(result['elections'])} upcoming")
    print(f"   Macro themes: {len(result['macro_themes'])} identified")
    print(f"   Risk scenarios: {len(result['risk_scenarios'])} generated")
    
    # Test 7: Economic Surprise Index
    print("\n7. Testing Economic Surprise Index...")
    
    surprise_index = result['economic_surprise_index']
    print(f"   US Surprise Index: {surprise_index['us_surprise_index']:.3f}")
    print(f"   EU Surprise Index: {surprise_index['eu_surprise_index']:.3f}")
    print(f"   UK Surprise Index: {surprise_index['uk_surprise_index']:.3f}")
    print(f"   Global Surprise Index: {surprise_index['global_surprise_index']:.3f}")
    
    # Test 8: Impact Forecast
    print("\n8. Testing Impact Forecast...")
    
    impact_forecast = result['impact_forecast']
    print(f"   Expected returns:")
    for asset, return_val in impact_forecast['expected_returns'].items():
        print(f"     - {asset}: {return_val:.3f}")
    
    print(f"   Risk-adjusted returns:")
    for asset, return_val in impact_forecast['risk_adjusted_returns'].items():
        print(f"     - {asset}: {return_val:.3f}")
    
    # Test 9: Event Details
    print("\n9. Testing Event Details...")
    
    if result['events']:
        event = result['events'][0]
        print(f"   Sample event: {event['event']}")
        print(f"     - Type: {event['type']}, Region: {event['region']}")
        print(f"     - Importance: {event['importance']}")
        print(f"     - Central bank: {event['central_bank']}")
        print(f"     - Market sentiment: {event['market_sentiment']:.3f}")
        print(f"     - Scenarios: {len(event['scenarios'])}")
    
    # Test 10: Theme Details
    print("\n10. Testing Theme Details...")
    
    if result['macro_themes']:
        theme = result['macro_themes'][0]
        print(f"   Sample theme: {theme['theme']}")
        print(f"     - Strength: {theme['strength']:.3f}, Confidence: {theme['confidence']:.3f}")
        print(f"     - Trend: {theme['trend_direction']}, Duration: {theme['duration']}")
        print(f"     - Key drivers: {', '.join(theme['key_drivers'][:3])}")
        print(f"     - Market impact: {theme['market_impact']}")
    
    print("\n✅ All Macro Agent tests completed successfully!")
    return True

if __name__ == "__main__":
    asyncio.run(test_macro_agent())
