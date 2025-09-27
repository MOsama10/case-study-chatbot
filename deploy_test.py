#!/usr/bin/env python3
"""
Deployment and comprehensive testing script for Case Study Chatbot PoC.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Any

def run_setup():
    """Run the setup script."""
    print("🔧 Running setup...")
    
    # Import and run setup
    try:
        from setup import main as setup_main
        if setup_main():
            print("✅ Setup completed successfully")
            return True
        else:
            print("❌ Setup failed")
            return False
    except Exception as e:
        print(f"❌ Setup error: {e}")
        return False

def test_embeddings():
    """Test embeddings functionality."""
    print("🔍 Testing embeddings...")
    
    try:
        from src.embeddings import get_embedding_manager, search_similar
        
        manager = get_embedding_manager()
        
        # Test queries
        test_queries = [
            "customer service",
            "employee training", 
            "cost reduction",
            "quality improvement",
            "problem solving"
        ]
        
        results_summary = []
        for query in test_queries:
            results = search_similar(query, top_k=3)
            results_summary.append({
                'query': query,
                'results_count': len(results),
                'top_score': results[0][1] if results else 0
            })
        
        print(f"  ✅ Embeddings working: {len(results_summary)} test queries completed")
        
        # Check quality
        avg_results = sum(r['results_count'] for r in results_summary) / len(results_summary)
        avg_score = sum(r['top_score'] for r in results_summary) / len(results_summary)
        
        print(f"  📊 Average results per query: {avg_results:.1f}")
        print(f"  📊 Average top score: {avg_score:.3f}")
        
        return avg_results >= 2
        
    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")
        return False

def test_knowledge_graph():
    """Test knowledge graph functionality."""
    print("🕸️ Testing knowledge graph...")
    
    try:
        from src.knowledge_graph import get_knowledge_graph, query_kg
        
        kg = get_knowledge_graph()
        
        # Basic structure tests
        node_count = kg.number_of_nodes()
        edge_count = kg.number_of_edges()
        
        print(f"  📊 KG structure: {node_count} nodes, {edge_count} edges")
        
        if node_count == 0:
            print("  ⚠️ Knowledge graph is empty")
            return False
        
        # Test queries
        test_queries = [
            "problem",
            "solution", 
            "customer",
            "training",
            "cost"
        ]
        
        query_results = []
        for query in test_queries:
            results = query_kg(query, kg, include_neighbors=True)
            query_results.append({
                'query': query,
                'results_count': len(results)
            })
        
        avg_kg_results = sum(r['results_count'] for r in query_results) / len(query_results)
        print(f"  📊 Average KG results per query: {avg_kg_results:.1f}")
        
        return avg_kg_results >= 1
        
    except Exception as e:
        print(f"❌ Knowledge graph test failed: {e}")
        return False

def test_retrieval_system():
    """Test the hybrid retrieval system."""
    print("🔄 Testing retrieval system...")
    
    try:
        from src.retriever import get_retriever, retrieve_for_query
        
        retriever = get_retriever()
        
        # Test different query types
        test_cases = [
            {
                'query': 'What are the main customer service problems?',
                'expected_intent': 'qa',
                'min_sources': 2
            },
            {
                'query': 'Analyze employee training effectiveness',
                'expected_intent': 'analysis',
                'min_sources': 3
            },
            {
                'query': 'Find cost reduction strategies',
                'expected_intent': 'search',
                'min_sources': 2
            },
            {
                'query': 'Recommend solutions for quality improvement',
                'expected_intent': 'recommendation',
                'min_sources': 2
            }
        ]
        
        passed_tests = 0
        for test_case in test_cases:
            try:
                result = retrieve_for_query(test_case['query'])
                
                # Check intent classification
                intent_correct = result['query_type'] == test_case['expected_intent']
                
                # Check source count
                sources_adequate = result['total_sources'] >= test_case['min_sources']
                
                # Check context quality
                context_adequate = len(result['context_text']) > 200
                
                if intent_correct and sources_adequate and context_adequate:
                    passed_tests += 1
                    print(f"  ✅ '{test_case['query'][:30]}...': {result['total_sources']} sources")
                else:
                    print(f"  ⚠️ '{test_case['query'][:30]}...': Issues detected")
                    if not intent_correct:
                        print(f"    - Intent: expected {test_case['expected_intent']}, got {result['query_type']}")
                    if not sources_adequate:
                        print(f"    - Sources: expected >={test_case['min_sources']}, got {result['total_sources']}")
                    if not context_adequate:
                        print(f"    - Context: too short ({len(result['context_text'])} chars)")
                        
            except Exception as e:
                print(f"  ❌ Query failed: {test_case['query'][:30]}... - {e}")
        
        success_rate = passed_tests / len(test_cases)
        print(f"  📊 Retrieval test success rate: {success_rate:.1%}")
        
        return success_rate >= 0.75  # 75% success rate required
        
    except Exception as e:
        print(f"❌ Retrieval system test failed: {e}")
        return False

def test_agent_responses():
    """Test the agent's response quality."""
    print("🤖 Testing agent responses...")
    
    try:
        from src.agent import get_agent
        
        agent = get_agent()
        
        # Test queries with expected characteristics
        test_queries = [
            {
                'query': 'What are common customer service issues?',
                'min_length': 100,
                'should_contain': ['customer', 'service'],
                'min_confidence': 0.3
            },
            {
                'query': 'How can companies reduce employee turnover?',
                'min_length': 150,
                'should_contain': ['employee', 'turnover'],
                'min_confidence': 0.3
            },
            {
                'query': 'Analyze the effectiveness of training programs',
                'min_length': 200,
                'should_contain': ['training', 'effectiveness'],
                'min_confidence': 0.4
            }
        ]
        
        passed_tests = 0
        total_response_time = 0
        
        for i, test_case in enumerate(test_queries, 1):
            try:
                print(f"  🧪 Test {i}: {test_case['query'][:40]}...")
                
                start_time = time.time()
                response = agent.answer_query(test_case['query'])
                response_time = time.time() - start_time
                total_response_time += response_time
                
                # Check response quality
                length_ok = len(response.answer) >= test_case['min_length']
                confidence_ok = response.confidence >= test_case['min_confidence']
                content_ok = all(term.lower() in response.answer.lower() 
                               for term in test_case['should_contain'])
                sources_ok = len(response.sources) > 0
                
                if length_ok and confidence_ok and content_ok and sources_ok:
                    passed_tests += 1
                    print(f"    ✅ Response quality good (confidence: {response.confidence:.2f}, time: {response_time:.2f}s)")
                else:
                    print(f"    ⚠️ Response quality issues:")
                    if not length_ok:
                        print(f"      - Length: {len(response.answer)} < {test_case['min_length']}")
                    if not confidence_ok:
                        print(f"      - Confidence: {response.confidence:.2f} < {test_case['min_confidence']}")
                    if not content_ok:
                        missing = [term for term in test_case['should_contain'] 
                                 if term.lower() not in response.answer.lower()]
                        print(f"      - Missing terms: {missing}")
                    if not sources_ok:
                        print(f"      - No sources provided")
                
            except Exception as e:
                print(f"    ❌ Agent test failed: {e}")
        
        avg_response_time = total_response_time / len(test_queries)
        success_rate = passed_tests / len(test_queries)
        
        print(f"  📊 Agent test success rate: {success_rate:.1%}")
        print(f"  📊 Average response time: {avg_response_time:.2f}s")
        
        return success_rate >= 0.67  # 67% success rate required
        
    except Exception as e:
        print(f"❌ Agent response test failed: {e}")
        return False

def test_ui_components():
    """Test UI components without launching the interface."""
    print("🖥️ Testing UI components...")
    
    try:
        from src.ui import ChatbotUI
        from src.chatbot import handle_message, get_session_info
        
        # Test chatbot manager
        test_user = "test_user_ui"
        test_message = "What are the main business challenges?"
        
        response = handle_message(test_user, test_message)
        
        # Check response structure
        required_keys = ['answer', 'sources', 'confidence', 'processing_time']
        structure_ok = all(key in response for key in required_keys)
        
        if structure_ok:
            print("  ✅ Chatbot manager working correctly")
        else:
            missing_keys = [key for key in required_keys if key not in response]
            print(f"  ⚠️ Missing response keys: {missing_keys}")
        
        # Test session info
        session_info = get_session_info(test_user)
        session_ok = session_info is not None and 'total_turns' in session_info
        
        if session_ok:
            print("  ✅ Session management working")
        else:
            print("  ⚠️ Session management issues")
        
        # Test UI initialization (without launching)
        ui = ChatbotUI()
        ui_ok = ui.setup_complete
        
        if ui_ok:
            print("  ✅ UI initialization successful")
        else:
            print("  ⚠️ UI initialization failed")
        
        return structure_ok and session_ok and ui_ok
        
    except Exception as e:
        print(f"❌ UI component test failed: {e}")
        return False

def generate_performance_report(test_results: Dict[str, bool]):
    """Generate a performance report."""
    print("\n📊 PERFORMANCE REPORT")
    print("=" * 50)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    success_rate = passed_tests / total_tests
    
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:25} {status}")
    
    print("-" * 50)
    print(f"Overall Success Rate: {success_rate:.1%} ({passed_tests}/{total_tests})")
    
    if success_rate >= 0.8:
        print("🎉 EXCELLENT: System is ready for production!")
    elif success_rate >= 0.6:
        print("✅ GOOD: System is functional with minor issues")
    elif success_rate >= 0.4:
        print("⚠️ FAIR: System needs improvement")
    else:
        print("❌ POOR: System requires significant fixes")
    
    return success_rate

def run_performance_benchmarks():
    """Run performance benchmarks."""
    print("\n⚡ PERFORMANCE BENCHMARKS")
    print("=" * 50)
    
    try:
        from src.agent import get_agent
        from src.retriever import get_retriever
        
        agent = get_agent()
        retriever = get_retriever()
        
        # Benchmark queries
        benchmark_queries = [
            "What are customer service problems?",
            "How to improve employee satisfaction?",
            "Cost reduction strategies",
            "Quality improvement methods",
            "Training program effectiveness"
        ]
        
        retrieval_times = []
        response_times = []
        
        for query in benchmark_queries:
            # Benchmark retrieval
            start = time.time()
            context = retriever.retrieve_context(query)
            retrieval_time = time.time() - start
            retrieval_times.append(retrieval_time)
            
            # Benchmark agent response
            start = time.time()
            response = agent.answer_query(query)
            response_time = time.time() - start
            response_times.append(response_time)
        
        avg_retrieval = sum(retrieval_times) / len(retrieval_times)
        avg_response = sum(response_times) / len(response_times)
        
        print(f"Average Retrieval Time: {avg_retrieval:.2f}s")
        print(f"Average Response Time:  {avg_response:.2f}s")
        print(f"Total Pipeline Time:    {avg_retrieval + avg_response:.2f}s")
        
        # Performance grades
        if avg_response < 5:
            print("🚀 FAST: Excellent response times")
        elif avg_response < 10:
            print("✅ GOOD: Acceptable response times")
        elif avg_response < 20:
            print("⚠️ SLOW: Consider optimization")
        else:
            print("❌ VERY SLOW: Requires optimization")
            
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")

def main():
    """Main deployment and testing function."""
    print("🚀 Case Study Chatbot PoC - Deployment & Testing")
    print("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Setup
    if not run_setup():
        print("❌ Setup failed. Cannot proceed with testing.")
        return False
    
    # Step 2: Run all tests
    test_results = {}
    
    test_results['Embeddings'] = test_embeddings()
    test_results['Knowledge Graph'] = test_knowledge_graph()
    test_results['Retrieval System'] = test_retrieval_system()
    test_results['Agent Responses'] = test_agent_responses()
    test_results['UI Components'] = test_ui_components()
    
    # Step 3: Generate report
    success_rate = generate_performance_report(test_results)
    
    # Step 4: Performance benchmarks
    if success_rate >= 0.6:  # Only run benchmarks if basic tests pass
        run_performance_benchmarks()
    
    # Step 5: Final recommendations
    total_time = time.time() - start_time
    
    print(f"\n⏱️ Total testing time: {total_time:.2f} seconds")
    
    if success_rate >= 0.8:
        print("\n🎉 READY TO LAUNCH!")
        print("Run the chatbot with: python src/ui.py")
    else:
        print("\n🔧 NEEDS ATTENTION")
        print("Please address the failed tests before deployment.")
        
        # Specific recommendations
        if not test_results.get('Embeddings', True):
            print("  - Check embedding model installation and API keys")
        if not test_results.get('Knowledge Graph', True):
            print("  - Verify document format and content structure")
        if not test_results.get('Retrieval System', True):
            print("  - Review retrieval configuration and thresholds")
        if not test_results.get('Agent Responses', True):
            print("  - Check LLM API connectivity and prompts")
        if not test_results.get('UI Components', True):
            print("  - Verify UI dependencies and configuration")
    
    return success_rate >= 0.6

if __name__ == "__main__":
    # Set up logging
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise during testing
    
    # Ensure we're in the right directory
    if not Path("src").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    success = main()
    sys.exit(0 if success else 1)