#!/usr/bin/env python3
"""
Diagnostic script to identify why the chatbot is giving generic responses.
Run this to debug the system: python diagnose.py
"""

import sys
import os
from pathlib import Path
import traceback

# Add src to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(project_root))

def test_component(name, test_func):
    """Test a component and report results."""
    print(f"\n🔍 Testing {name}...")
    try:
        result = test_func()
        if result:
            print(f"✅ {name}: OK")
            return True
        else:
            print(f"⚠️ {name}: Working but with issues")
            return True
    except Exception as e:
        print(f"❌ {name}: FAILED")
        print(f"   Error: {e}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def test_api_connection():
    """Test API connection."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        llm_provider = os.getenv("LLM_PROVIDER", "gemini")
        
        print(f"   LLM Provider: {llm_provider}")
        print(f"   Gemini Key: {'✅ Set' if gemini_key and gemini_key != 'your_gemini_api_key_here' else '❌ Not set'}")
        print(f"   OpenAI Key: {'✅ Set' if openai_key and openai_key != 'your_openai_api_key_here' else '❌ Not set'}")
        
        if llm_provider == "gemini" and gemini_key and gemini_key != "your_gemini_api_key_here":
            # Test Gemini connection
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content("Test message: What is 2+2?")
            print(f"   Gemini Test: ✅ {response.text[:50]}...")
            return True
        elif llm_provider == "openai" and openai_key and openai_key != "your_openai_api_key_here":
            # Test OpenAI connection
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Test: What is 2+2?"}],
                max_tokens=50
            )
            print(f"   OpenAI Test: ✅ {response.choices[0].message.content[:50]}...")
            return True
        else:
            print("   ❌ No valid API key configured")
            return False
            
    except Exception as e:
        print(f"   ❌ API connection failed: {e}")
        return False

def test_embeddings():
    """Test embeddings system."""
    try:
        import src.embeddings as emb
        manager = emb.get_embedding_manager()
        
        # Test if embeddings are working
        test_texts = ["arbitration procedure", "legal costs", "contract validity"]
        embeddings = manager.generate_embeddings(test_texts)
        print(f"   Generated embeddings shape: {embeddings.shape}")
        
        # Test if index exists
        if manager.metadata:
            print(f"   Index has {len(manager.metadata)} items")
            
            # Test search
            results = manager.search("arbitration costs", top_k=3)
            print(f"   Search test: Found {len(results)} results")
            if results:
                print(f"   Top result: {results[0][2][:100]}...")
            return True
        else:
            print("   ⚠️ No index built yet")
            return False
            
    except Exception as e:
        print(f"   ❌ Embeddings failed: {e}")
        return False

def test_knowledge_graph():
    """Test knowledge graph."""
    try:
        import src.knowledge_graph as kg
        graph = kg.get_knowledge_graph()
        
        print(f"   KG has {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Test query
        results = kg.query_kg("arbitration", graph)
        print(f"   KG search test: Found {len(results)} results")
        
        return True
    except Exception as e:
        print(f"   ❌ Knowledge graph failed: {e}")
        return False

def test_data_loading():
    """Test data loading."""
    try:
        import src.data_loader as dl
        
        data_dir = project_root / "data"
        summary = dl.get_document_summary(data_dir)
        
        print(f"   Found {summary['total_documents']} documents")
        print(f"   Document types: {[doc['type'] for doc in summary['documents']]}")
        
        if summary['total_documents'] > 0:
            # Try loading documents
            items = dl.load_multiple_docx(data_dir)
            print(f"   Loaded {len(items)} text items")
            if items:
                print(f"   Sample item: {items[0]['text'][:100]}...")
            return True
        else:
            print("   ⚠️ No documents found")
            return False
            
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        return False

def test_retriever():
    """Test retriever system."""
    try:
        import src.retriever as ret
        retriever = ret.get_retriever()
        
        # Test retrieval
        context = ret.retrieve_for_query("What are arbitration costs?")
        print(f"   Retrieved {context.get('total_sources', 0)} sources")
        print(f"   Query type: {context.get('query_type', 'unknown')}")
        
        if context.get('sources'):
            print(f"   Sample source: {context['sources'][0]['text'][:100]}...")
        
        return True
    except Exception as e:
        print(f"   ❌ Retriever failed: {e}")
        return False

def test_agent():
    """Test agent system."""
    try:
        import src.agent as agent_module
        agent = agent_module.get_agent()
        
        # Test query
        response = agent_module.answer_query("What is arbitration?")
        
        print(f"   Response length: {len(response['answer'])} chars")
        print(f"   Confidence: {response['confidence']:.2f}")
        print(f"   Query type: {response['query_type']}")
        print(f"   Sources: {len(response['sources'])}")
        print(f"   Response preview: {response['answer'][:150]}...")
        
        # Check if it's a real response or fallback
        if "technical difficulties" in response['answer'] or "mock response" in response['answer']:
            print("   ⚠️ Agent is in fallback mode")
            return False
        else:
            print("   ✅ Agent providing real responses")
            return True
            
    except Exception as e:
        print(f"   ❌ Agent failed: {e}")
        return False

def test_full_pipeline():
    """Test the complete pipeline."""
    try:
        import src.chatbot as chatbot
        
        # Test handle_message function
        response = chatbot.handle_message("test_user", "What are the costs of arbitration?")
        
        print(f"   Pipeline response length: {len(response['answer'])} chars")
        print(f"   Pipeline confidence: {response['confidence']:.2f}")
        print(f"   Pipeline response: {response['answer'][:150]}...")
        
        # Check if real response
        if ("technical difficulties" in response['answer'] or 
            "mock response" in response['answer'] or
            "fallback mode" in response['answer']):
            print("   ⚠️ Pipeline is in fallback mode")
            return False
        else:
            print("   ✅ Pipeline working correctly")
            return True
            
    except Exception as e:
        print(f"   ❌ Pipeline failed: {e}")
        return False

def main():
    """Run complete diagnostic."""
    print("🔍 LEGAL DOCUMENT AI ASSISTANT - SYSTEM DIAGNOSTIC")
    print("=" * 60)
    
    # Test all components
    results = {}
    
    results['api'] = test_component("API Connection", test_api_connection)
    results['data'] = test_component("Data Loading", test_data_loading)
    results['embeddings'] = test_component("Embeddings System", test_embeddings)
    results['kg'] = test_component("Knowledge Graph", test_knowledge_graph)
    results['retriever'] = test_component("Retriever System", test_retriever)
    results['agent'] = test_component("Agent System", test_agent)
    results['pipeline'] = test_component("Full Pipeline", test_full_pipeline)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    working = sum(results.values())
    total = len(results)
    
    for component, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {component.title()}: {'OK' if status else 'FAILED'}")
    
    print(f"\nOverall: {working}/{total} components working")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if not results['api']:
        print("1. 🔑 Configure your API key in .env file")
        print("   - Edit .env and add GEMINI_API_KEY or OPENAI_API_KEY")
    
    if not results['data']:
        print("2. 📚 Add document data")
        print("   - Add .docx or .txt files to data/ directory")
    
    if not results['embeddings']:
        print("3. 🔍 Rebuild embeddings index")
        print("   - Delete storage/ folder and restart")
    
    if not results['agent']:
        print("4. 🤖 Check agent configuration")
        print("   - Verify LLM provider settings")
    
    if working == total:
        print("🎉 All systems working! The chatbot should provide proper responses.")
    elif working >= 4:
        print("⚠️ Most systems working. Check the failed components above.")
    else:
        print("❌ Multiple system failures. Follow recommendations above.")

if __name__ == "__main__":
    main()