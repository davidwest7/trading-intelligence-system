#!/usr/bin/env python3
"""
Comprehensive System Validation
Validates all agents, data adapters, and data points across the entire solution
"""
import asyncio
import json
import time
import importlib
import inspect
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

class ComprehensiveSystemValidator:
    def __init__(self):
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL']
        self.validation_results = {}
        self.agent_results = {}
        self.adapter_results = {}
        self.system_components = {}
        
    def discover_agents(self) -> Dict[str, Any]:
        """Discover all available agents in the system"""
        agents_dir = Path('agents')
        discovered_agents = {}
        
        if not agents_dir.exists():
            return discovered_agents
        
        # Discover agent directories
        for agent_dir in agents_dir.iterdir():
            if agent_dir.is_dir() and not agent_dir.name.startswith('__'):
                agent_name = agent_dir.name
                discovered_agents[agent_name] = {
                    'path': str(agent_dir),
                    'files': [],
                    'main_agent': None,
                    'models': None
                }
                
                # Look for agent files
                for file_path in agent_dir.glob('*.py'):
                    if not file_path.name.startswith('__'):
                        discovered_agents[agent_name]['files'].append(file_path.name)
                        
                        # Try to identify main agent file
                        if 'agent' in file_path.name.lower() and 'main' not in file_path.name.lower():
                            discovered_agents[agent_name]['main_agent'] = file_path.name
                        
                        # Look for models file
                        if 'models' in file_path.name.lower():
                            discovered_agents[agent_name]['models'] = file_path.name
        
        return discovered_agents
    
    def discover_data_adapters(self) -> Dict[str, Any]:
        """Discover all data adapters in the system"""
        adapters_dir = Path('common/data_adapters')
        discovered_adapters = {}
        
        if not adapters_dir.exists():
            return discovered_adapters
        
        # Discover adapter files
        for adapter_file in adapters_dir.glob('*.py'):
            if not adapter_file.name.startswith('__'):
                adapter_name = adapter_file.stem
                discovered_adapters[adapter_name] = {
                    'path': str(adapter_file),
                    'filename': adapter_file.name,
                    'class_name': None
                }
        
        return discovered_adapters
    
    def discover_system_components(self) -> Dict[str, Any]:
        """Discover all system components"""
        components = {
            'common_modules': {},
            'evaluation_modules': {},
            'feature_store': {},
            'opportunity_store': {},
            'unified_scorers': {}
        }
        
        # Common modules
        common_dir = Path('common')
        if common_dir.exists():
            for file_path in common_dir.glob('*.py'):
                if not file_path.name.startswith('__'):
                    components['common_modules'][file_path.stem] = {
                        'path': str(file_path),
                        'filename': file_path.name
                    }
        
        # Evaluation modules
        eval_dir = Path('common/evaluation')
        if eval_dir.exists():
            for file_path in eval_dir.glob('*.py'):
                if not file_path.name.startswith('__'):
                    components['evaluation_modules'][file_path.stem] = {
                        'path': str(file_path),
                        'filename': file_path.name
                    }
        
        # Feature store
        feature_dir = Path('common/feature_store')
        if feature_dir.exists():
            for file_path in feature_dir.glob('*.py'):
                if not file_path.name.startswith('__'):
                    components['feature_store'][file_path.stem] = {
                        'path': str(file_path),
                        'filename': file_path.name
                    }
        
        return components
    
    async def validate_agent(self, agent_name: str, agent_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single agent"""
        print(f"\n🤖 Validating Agent: {agent_name}")
        print("=" * 50)
        
        validation_result = {
            'agent_name': agent_name,
            'valid': True,
            'files_found': len(agent_info['files']),
            'main_agent_file': agent_info['main_agent'],
            'models_file': agent_info['models'],
            'import_success': False,
            'class_discovery': False,
            'methods_found': [],
            'errors': [],
            'warnings': []
        }
        
        try:
            # Try to import the agent module
            if agent_info['main_agent']:
                module_path = f"agents.{agent_name}.{agent_info['main_agent'].replace('.py', '')}"
                try:
                    module = importlib.import_module(module_path)
                    validation_result['import_success'] = True
                    
                    # Discover classes and methods
                    classes = inspect.getmembers(module, inspect.isclass)
                    for class_name, class_obj in classes:
                        if 'Agent' in class_name:
                            validation_result['class_discovery'] = True
                            validation_result['methods_found'] = [method for method in dir(class_obj) 
                                                                if not method.startswith('_') and callable(getattr(class_obj, method))]
                            break
                    
                except ImportError as e:
                    validation_result['valid'] = False
                    validation_result['errors'].append(f"Import error: {str(e)}")
                except Exception as e:
                    validation_result['warnings'].append(f"Module analysis error: {str(e)}")
            
            # Check file structure
            if not agent_info['files']:
                validation_result['warnings'].append("No Python files found in agent directory")
            
            if not agent_info['main_agent']:
                validation_result['warnings'].append("No main agent file identified")
            
            # Print results
            print(f"📁 Files Found: {validation_result['files_found']}")
            print(f"🤖 Main Agent: {validation_result['main_agent_file'] or 'Not found'}")
            print(f"📊 Models: {validation_result['models_file'] or 'Not found'}")
            print(f"📦 Import Success: {'✅ Yes' if validation_result['import_success'] else '❌ No'}")
            print(f"🔍 Class Discovery: {'✅ Yes' if validation_result['class_discovery'] else '❌ No'}")
            
            if validation_result['methods_found']:
                print(f"⚙️ Methods Found: {len(validation_result['methods_found'])}")
                for method in validation_result['methods_found'][:5]:  # Show first 5
                    print(f"   • {method}")
                if len(validation_result['methods_found']) > 5:
                    print(f"   ... and {len(validation_result['methods_found']) - 5} more")
            
            if validation_result['errors']:
                print(f"❌ Errors: {len(validation_result['errors'])}")
                for error in validation_result['errors']:
                    print(f"   • {error}")
            
            if validation_result['warnings']:
                print(f"⚠️ Warnings: {len(validation_result['warnings'])}")
                for warning in validation_result['warnings']:
                    print(f"   • {warning}")
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
            print(f"❌ Validation error: {str(e)}")
        
        return validation_result
    
    async def validate_data_adapter(self, adapter_name: str, adapter_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single data adapter"""
        print(f"\n🔌 Validating Data Adapter: {adapter_name}")
        print("=" * 50)
        
        validation_result = {
            'adapter_name': adapter_name,
            'valid': True,
            'filename': adapter_info['filename'],
            'import_success': False,
            'class_discovery': False,
            'methods_found': [],
            'api_key_required': False,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Try to import the adapter module
            module_path = f"common.data_adapters.{adapter_name}"
            try:
                module = importlib.import_module(module_path)
                validation_result['import_success'] = True
                
                # Discover classes and methods
                classes = inspect.getmembers(module, inspect.isclass)
                for class_name, class_obj in classes:
                    if any(keyword in class_name.lower() for keyword in ['adapter', 'client', 'api']):
                        validation_result['class_discovery'] = True
                        validation_result['methods_found'] = [method for method in dir(class_obj) 
                                                            if not method.startswith('_') and callable(getattr(class_obj, method))]
                        
                        # Check if API key is required
                        if hasattr(class_obj, '__init__'):
                            init_sig = inspect.signature(class_obj.__init__)
                            for param_name, param in init_sig.parameters.items():
                                if 'key' in param_name.lower() or 'token' in param_name.lower():
                                    validation_result['api_key_required'] = True
                                    break
                        break
                
            except ImportError as e:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Import error: {str(e)}")
            except Exception as e:
                validation_result['warnings'].append(f"Module analysis error: {str(e)}")
            
            # Print results
            print(f"📁 File: {validation_result['filename']}")
            print(f"📦 Import Success: {'✅ Yes' if validation_result['import_success'] else '❌ No'}")
            print(f"🔍 Class Discovery: {'✅ Yes' if validation_result['class_discovery'] else '❌ No'}")
            print(f"🔑 API Key Required: {'✅ Yes' if validation_result['api_key_required'] else '❌ No'}")
            
            if validation_result['methods_found']:
                print(f"⚙️ Methods Found: {len(validation_result['methods_found'])}")
                for method in validation_result['methods_found'][:5]:  # Show first 5
                    print(f"   • {method}")
                if len(validation_result['methods_found']) > 5:
                    print(f"   ... and {len(validation_result['methods_found']) - 5} more")
            
            if validation_result['errors']:
                print(f"❌ Errors: {len(validation_result['errors'])}")
                for error in validation_result['errors']:
                    print(f"   • {error}")
            
            if validation_result['warnings']:
                print(f"⚠️ Warnings: {len(validation_result['warnings'])}")
                for warning in validation_result['warnings']:
                    print(f"   • {warning}")
        
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
            print(f"❌ Validation error: {str(e)}")
        
        return validation_result
    
        async def validate_system_component(self, component_name: str, component_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a system component"""
        print(f"\n🔧 Validating System Component: {component_name}")
        print("=" * 50)
        
        validation_result = {
            'component_name': component_name,
            'valid': True,
            'filename': component_info['filename'],
            'import_success': False,
            'class_discovery': False,
            'methods_found': [],
            'errors': [],
            'warnings': []
        }
        
        try:
            # Try to import the component module with correct path structure
            if component_name.startswith('evaluation_'):
                # Handle evaluation modules
                actual_name = component_name.replace('evaluation_', '')
                module_path = f"common.evaluation.{actual_name}"
            elif component_name.startswith('feature_store_'):
                # Handle feature store modules
                actual_name = component_name.replace('feature_store_', '')
                module_path = f"common.feature_store.{actual_name}"
            else:
                # Handle other common modules
                module_path = f"common.{component_name}"
            
            module = importlib.import_module(module_path)
            validation_result['import_success'] = True
            
            # Discover classes and methods
            classes = inspect.getmembers(module, inspect.isclass)
            for class_name, class_obj in classes:
                validation_result['class_discovery'] = True
                validation_result['methods_found'] = [method for method in dir(class_obj) 
                                                    if not method.startswith('_') and callable(getattr(class_obj, method))]
                break
            
        except ImportError as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Import error: {str(e)}")
        except Exception as e:
            validation_result['warnings'].append(f"Module analysis error: {str(e)}")
        
        # Print results
        print(f"📁 File: {validation_result['filename']}")
        print(f"📦 Import Success: {'✅ Yes' if validation_result['import_success'] else '❌ No'}")
        print(f"🔍 Class Discovery: {'✅ Yes' if validation_result['class_discovery'] else '❌ No'}")
        
        if validation_result['methods_found']:
            print(f"⚙️ Methods Found: {len(validation_result['methods_found'])}")
            for method in validation_result['methods_found'][:5]:  # Show first 5
                print(f"   • {method}")
            if len(validation_result['methods_found']) > 5:
                print(f"   ... and {len(validation_result['methods_found']) - 5} more")
        
        if validation_result['errors']:
            print(f"❌ Errors: {len(validation_result['errors'])}")
            for error in validation_result['errors']:
                print(f"   • {error}")
        
        if validation_result['warnings']:
            print(f"⚠️ Warnings: {len(validation_result['warnings'])}")
            for warning in validation_result['warnings']:
                print(f"   • {warning}")
        
        return validation_result
    
    async def test_data_integration_workflow(self) -> Dict[str, Any]:
        """Test the complete data integration workflow"""
        print(f"\n🔄 Testing Data Integration Workflow")
        print("=" * 60)
        
        try:
            from comprehensive_data_integration_phase4 import ComprehensiveDataIntegrationPhase4
            
            integration = ComprehensiveDataIntegrationPhase4()
            workflow_result = {
                'valid': True,
                'symbols_tested': [],
                'data_sources_working': {},
                'collection_times': [],
                'errors': [],
                'warnings': []
            }
            
            for symbol in self.test_symbols:
                print(f"\n📊 Testing {symbol}...")
                try:
                    start_time = time.time()
                    data = await integration.get_comprehensive_data(symbol)
                    collection_time = time.time() - start_time
                    
                    workflow_result['symbols_tested'].append(symbol)
                    workflow_result['collection_times'].append(collection_time)
                    
                    # Check data sources
                    sources = data.get('sources', {})
                    for source_name, source_data in sources.items():
                        if source_name not in workflow_result['data_sources_working']:
                            workflow_result['data_sources_working'][source_name] = {'working': 0, 'total': 0}
                        
                        workflow_result['data_sources_working'][source_name]['total'] += 1
                        if source_data.get('status') == 'WORKING':
                            workflow_result['data_sources_working'][source_name]['working'] += 1
                    
                    print(f"   ✅ {symbol}: {collection_time:.2f}s")
                    
                except Exception as e:
                    workflow_result['errors'].append(f"Error testing {symbol}: {str(e)}")
                    print(f"   ❌ {symbol}: Error - {str(e)}")
            
            # Calculate success rates
            for source_name, stats in workflow_result['data_sources_working'].items():
                if stats['total'] > 0:
                    stats['success_rate'] = (stats['working'] / stats['total']) * 100
                else:
                    stats['success_rate'] = 0
            
            # Print workflow results
            print(f"\n📊 WORKFLOW RESULTS:")
            print(f"   Symbols Tested: {len(workflow_result['symbols_tested'])}")
            print(f"   Average Collection Time: {sum(workflow_result['collection_times']) / len(workflow_result['collection_times']):.2f}s")
            
            print(f"\n📡 DATA SOURCE SUCCESS RATES:")
            for source_name, stats in workflow_result['data_sources_working'].items():
                status_emoji = '✅' if stats['success_rate'] >= 80 else '⚠️' if stats['success_rate'] >= 50 else '❌'
                print(f"   {status_emoji} {source_name.upper()}: {stats['success_rate']:.1f}% ({stats['working']}/{stats['total']})")
            
            if workflow_result['errors']:
                print(f"\n❌ Workflow Errors: {len(workflow_result['errors'])}")
                for error in workflow_result['errors']:
                    print(f"   • {error}")
            
            return workflow_result
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Workflow test error: {str(e)}"]
            }
    
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation of the entire system"""
        print("🚀 Comprehensive System Validation")
        print("=" * 60)
        print(f"📅 Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Testing: All Agents, Data Adapters, and System Components")
        print("=" * 60)
        
        start_time = time.time()
        
        # 1. Discover system components
        print("\n🔍 DISCOVERING SYSTEM COMPONENTS...")
        agents = self.discover_agents()
        adapters = self.discover_data_adapters()
        components = self.discover_system_components()
        
        print(f"🤖 Agents Found: {len(agents)}")
        print(f"🔌 Data Adapters Found: {len(adapters)}")
        print(f"🔧 System Components Found: {sum(len(comp) for comp in components.values())}")
        
        # 2. Validate agents
        print(f"\n🤖 VALIDATING AGENTS ({len(agents)} total)...")
        for agent_name, agent_info in agents.items():
            result = await self.validate_agent(agent_name, agent_info)
            self.agent_results[agent_name] = result
        
        # 3. Validate data adapters
        print(f"\n🔌 VALIDATING DATA ADAPTERS ({len(adapters)} total)...")
        for adapter_name, adapter_info in adapters.items():
            result = await self.validate_data_adapter(adapter_name, adapter_info)
            self.adapter_results[adapter_name] = result
        
        # 4. Validate system components
        print(f"\n🔧 VALIDATING SYSTEM COMPONENTS...")
        for component_type, component_list in components.items():
            print(f"\n📁 {component_type.upper()}:")
            for component_name, component_info in component_list.items():
                result = await self.validate_system_component(component_name, component_info)
                self.system_components[f"{component_type}_{component_name}"] = result
        
        # 5. Test data integration workflow
        workflow_result = await self.test_data_integration_workflow()
        
        total_time = time.time() - start_time
        
        # Generate comprehensive report
        report = self._generate_comprehensive_report(total_time, workflow_result)
        
        return report
    
    def _generate_comprehensive_report(self, total_time: float, workflow_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive system validation report"""
        print(f"\n📋 COMPREHENSIVE SYSTEM VALIDATION REPORT")
        print("=" * 60)
        
        # Calculate statistics
        total_agents = len(self.agent_results)
        valid_agents = sum(1 for r in self.agent_results.values() if r['valid'])
        invalid_agents = total_agents - valid_agents
        
        total_adapters = len(self.adapter_results)
        valid_adapters = sum(1 for r in self.adapter_results.values() if r['valid'])
        invalid_adapters = total_adapters - valid_adapters
        
        total_components = len(self.system_components)
        valid_components = sum(1 for r in self.system_components.values() if r['valid'])
        invalid_components = total_components - valid_components
        
        # Count total errors and warnings
        total_errors = (
            sum(len(r.get('errors', [])) for r in self.agent_results.values()) +
            sum(len(r.get('errors', [])) for r in self.adapter_results.values()) +
            sum(len(r.get('errors', [])) for r in self.system_components.values()) +
            len(workflow_result.get('errors', []))
        )
        
        total_warnings = (
            sum(len(r.get('warnings', [])) for r in self.agent_results.values()) +
            sum(len(r.get('warnings', [])) for r in self.adapter_results.values()) +
            sum(len(r.get('warnings', [])) for r in self.system_components.values()) +
            len(workflow_result.get('warnings', []))
        )
        
        # Print summary
        print(f"📊 SYSTEM VALIDATION STATISTICS:")
        print(f"   Total Validation Time: {total_time:.2f}s")
        print(f"   Total Errors: {total_errors}")
        print(f"   Total Warnings: {total_warnings}")
        
        print(f"\n🤖 AGENT VALIDATION:")
        print(f"   Total Agents: {total_agents}")
        print(f"   Valid Agents: {valid_agents}")
        print(f"   Invalid Agents: {invalid_agents}")
        print(f"   Agent Success Rate: {(valid_agents/total_agents*100):.1f}%" if total_agents > 0 else "   Agent Success Rate: N/A")
        
        print(f"\n🔌 DATA ADAPTER VALIDATION:")
        print(f"   Total Adapters: {total_adapters}")
        print(f"   Valid Adapters: {valid_adapters}")
        print(f"   Invalid Adapters: {invalid_adapters}")
        print(f"   Adapter Success Rate: {(valid_adapters/total_adapters*100):.1f}%" if total_adapters > 0 else "   Adapter Success Rate: N/A")
        
        print(f"\n🔧 SYSTEM COMPONENT VALIDATION:")
        print(f"   Total Components: {total_components}")
        print(f"   Valid Components: {valid_components}")
        print(f"   Invalid Components: {invalid_components}")
        print(f"   Component Success Rate: {(valid_components/total_components*100):.1f}%" if total_components > 0 else "   Component Success Rate: N/A")
        
        # Detailed agent results
        print(f"\n📋 DETAILED AGENT RESULTS:")
        for agent_name, result in self.agent_results.items():
            status_emoji = '✅' if result['valid'] else '❌'
            print(f"   {status_emoji} {agent_name}: {'Valid' if result['valid'] else 'Invalid'}")
            if result['errors']:
                for error in result['errors']:
                    print(f"      • {error}")
        
        # Detailed adapter results
        print(f"\n📋 DETAILED ADAPTER RESULTS:")
        for adapter_name, result in self.adapter_results.items():
            status_emoji = '✅' if result['valid'] else '❌'
            print(f"   {status_emoji} {adapter_name}: {'Valid' if result['valid'] else 'Invalid'}")
            if result['errors']:
                for error in result['errors']:
                    print(f"      • {error}")
        
        # Workflow results
        if workflow_result.get('valid', False):
            print(f"\n🔄 DATA INTEGRATION WORKFLOW:")
            print(f"   Symbols Tested: {len(workflow_result['symbols_tested'])}")
            if workflow_result['collection_times']:
                avg_time = sum(workflow_result['collection_times']) / len(workflow_result['collection_times'])
                print(f"   Average Collection Time: {avg_time:.2f}s")
            
            print(f"   Data Source Success Rates:")
            for source_name, stats in workflow_result.get('data_sources_working', {}).items():
                status_emoji = '✅' if stats.get('success_rate', 0) >= 80 else '⚠️' if stats.get('success_rate', 0) >= 50 else '❌'
                print(f"     {status_emoji} {source_name.upper()}: {stats.get('success_rate', 0):.1f}%")
        
        # Recommendations
        print(f"\n💡 SYSTEM VALIDATION RECOMMENDATIONS:")
        
        overall_success = (
            (valid_agents / total_agents if total_agents > 0 else 1) +
            (valid_adapters / total_adapters if total_adapters > 0 else 1) +
            (valid_components / total_components if total_components > 0 else 1)
        ) / 3 * 100
        
        if overall_success >= 90 and total_errors == 0:
            print("   ✅ System is in excellent condition - ready for production")
        elif overall_success >= 80 and total_errors == 0:
            print("   ⚠️ System is mostly valid - review warnings for improvements")
        elif overall_success >= 70:
            print("   ⚠️ System has some issues - review errors and fix critical problems")
        else:
            print("   ❌ System has significant issues - requires immediate attention")
        
        if total_errors > 0:
            print(f"   🔧 Fix {total_errors} errors before production deployment")
        
        if total_warnings > 0:
            print(f"   📝 Review {total_warnings} warnings for potential improvements")
        
        # Create report object
        report = {
            'validation_date': datetime.now().isoformat(),
            'total_validation_time': total_time,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'agent_results': self.agent_results,
            'adapter_results': self.adapter_results,
            'system_components': self.system_components,
            'workflow_result': workflow_result,
            'overall_success_rate': overall_success,
            'recommendations': self._generate_system_recommendations(
                valid_agents, total_agents, valid_adapters, total_adapters, 
                valid_components, total_components, total_errors, total_warnings
            )
        }
        
        return report
    
    def _generate_system_recommendations(self, valid_agents, total_agents, valid_adapters, total_adapters, 
                                       valid_components, total_components, total_errors, total_warnings) -> List[str]:
        """Generate system-wide recommendations"""
        recommendations = []
        
        # Overall system health
        overall_success = (
            (valid_agents / total_agents if total_agents > 0 else 1) +
            (valid_adapters / total_adapters if total_adapters > 0 else 1) +
            (valid_components / total_components if total_components > 0 else 1)
        ) / 3 * 100
        
        if overall_success >= 90 and total_errors == 0:
            recommendations.append("✅ System is in excellent condition - ready for production deployment")
        elif overall_success >= 80 and total_errors == 0:
            recommendations.append("⚠️ System is mostly valid - review warnings for optimal performance")
        elif overall_success >= 70:
            recommendations.append("⚠️ System has some issues - review errors and fix critical problems")
        else:
            recommendations.append("❌ System has significant issues - requires immediate attention")
        
        # Agent-specific recommendations
        if total_agents > 0:
            agent_success_rate = (valid_agents / total_agents) * 100
            if agent_success_rate < 80:
                recommendations.append(f"🔧 Review and fix {total_agents - valid_agents} agent issues")
            elif agent_success_rate < 100:
                recommendations.append("📝 Review agent warnings for potential improvements")
        
        # Adapter-specific recommendations
        if total_adapters > 0:
            adapter_success_rate = (valid_adapters / total_adapters) * 100
            if adapter_success_rate < 80:
                recommendations.append(f"🔧 Review and fix {total_adapters - valid_adapters} adapter issues")
            elif adapter_success_rate < 100:
                recommendations.append("📝 Review adapter warnings for potential improvements")
        
        # Component-specific recommendations
        if total_components > 0:
            component_success_rate = (valid_components / total_components) * 100
            if component_success_rate < 80:
                recommendations.append(f"🔧 Review and fix {total_components - valid_components} component issues")
            elif component_success_rate < 100:
                recommendations.append("📝 Review component warnings for potential improvements")
        
        # Error and warning recommendations
        if total_errors > 0:
            recommendations.append(f"🔧 Fix {total_errors} errors before production deployment")
        
        if total_warnings > 0:
            recommendations.append(f"📝 Address {total_warnings} warnings for optimal performance")
        
        return recommendations
    
    async def save_validation_report(self, report: Dict[str, Any], filename: str = None):
        """Save comprehensive validation report to file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"comprehensive_system_validation_report_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n💾 Comprehensive validation report saved to: {filename}")
        except Exception as e:
            print(f"❌ Failed to save validation report: {str(e)}")

async def main():
    """Run comprehensive system validation"""
    print("🚀 Starting Comprehensive System Validation")
    print("=" * 60)
    
    # Create validator instance
    validator = ComprehensiveSystemValidator()
    
    # Run comprehensive validation
    report = await validator.run_comprehensive_validation()
    
    # Save report
    await validator.save_validation_report(report)
    
    # Final summary
    print(f"\n🎉 COMPREHENSIVE SYSTEM VALIDATION COMPLETE!")
    print(f"📊 Overall Success Rate: {report['overall_success_rate']:.1f}%")
    print(f"❌ Total Errors: {report['total_errors']}")
    print(f"⚠️ Total Warnings: {report['total_warnings']}")
    print(f"⏱️ Total Time: {report['total_validation_time']:.2f}s")
    
    if report['overall_success_rate'] >= 90 and report['total_errors'] == 0:
        print("✅ System is in excellent condition - ready for production!")
    elif report['overall_success_rate'] >= 80 and report['total_errors'] == 0:
        print("⚠️ System is mostly valid - review warnings for improvements")
    elif report['overall_success_rate'] >= 70:
        print("⚠️ System has some issues - review errors and fix critical problems")
    else:
        print("❌ System has significant issues - requires immediate attention")

if __name__ == "__main__":
    asyncio.run(main())
