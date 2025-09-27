# #!/usr/bin/env python3
# """
# Simple launcher script for the Case Study Chatbot.
# This script handles all setup and launches the enhanced UI.
# """

# import sys
# import os
# import subprocess
# import time
# from pathlib import Path
# import importlib.util

# def print_banner():
#     """Print startup banner."""
#     print("""
# 🚀 Case Study Chatbot - Enhanced AI Business Analyzer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 🧠 AI-Powered Analysis  📊 Vector Search  🕸️ Knowledge Graph
#     """)

# def check_requirements():
#     """Check if basic requirements are met."""
#     print("🔍 Checking system requirements...")
    
#     # Check Python version
#     if sys.version_info < (3, 10):
#         print(f"❌ Python 3.10+ required, found {sys.version_info.major}.{sys.version_info.minor}")
#         print("💡 Please upgrade Python: https://www.python.org/downloads/")
#         return False
    
#     print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} OK")
    
#     # Check if in correct directory
#     if not Path("src").exists():
#         print("❌ Please run this from the project root directory (where src/ folder is)")
#         print(f"📁 Current directory: {Path.cwd()}")
#         print("💡 Navigate to your case-study-chatbot directory first")
#         return False
    
#     print("✅ Project structure OK")
#     return True

# def install_dependencies():
#     """Install required packages."""
#     print("📦 Installing/checking dependencies...")
    
#     required_packages = [
#         "gradio>=3.50.0",
#         "python-dotenv>=1.0.0",
#         "sentence-transformers>=2.2.2",
#         "faiss-cpu>=1.7.4",
#         "networkx>=3.1.0",
#         "python-docx>=0.8.11",
#         "google-generativeai>=0.3.0",
#         "openai>=1.3.0",
#         "numpy>=1.24.0",
#         "pandas>=2.0.0",
#         "requests>=2.31.0",
#         "tqdm>=4.65.0",
#         "rich>=13.5.0"
#     ]
    
#     missing_packages = []
    
#     for package in required_packages:
#         try:
#             # Extract package name (before >= or ==)
#             package_name = package.split(">=")[0].split("==")[0]
            
#             # Handle special cases for import names
#             import_name = package_name
#             if package_name == "python-dotenv":
#                 import_name = "dotenv"
#             elif package_name == "python-docx":
#                 import_name = "docx"
#             elif package_name == "faiss-cpu":
#                 import_name = "faiss"
#             elif package_name == "google-generativeai":
#                 import_name = "google.generativeai"
#             elif package_name == "sentence-transformers":
#                 import_name = "sentence_transformers"
            
#             # Try to import
#             if "." in import_name:
#                 # Handle nested imports like google.generativeai
#                 main_module = import_name.split(".")[0]
#                 __import__(main_module)
#             else:
#                 __import__(import_name)
                
#         except ImportError:
#             missing_packages.append(package)
    
#     if missing_packages:
#         print(f"⚠️ Installing {len(missing_packages)} missing packages...")
#         for package in missing_packages:
#             try:
#                 print(f"   Installing {package}...")
#                 subprocess.check_call([
#                     sys.executable, "-m", "pip", "install", package, "--quiet"
#                 ], stdout=subprocess.DEVNULL)
#             except subprocess.CalledProcessError as e:
#                 print(f"   ⚠️ Failed to install {package}: {e}")
    
#     print("✅ Dependencies ready")

# def setup_environment():
#     """Set up the environment."""
#     print("🔧 Setting up environment...")
    
#     # Create .env file if it doesn't exist
#     env_file = Path(".env")
#     if not env_file.exists():
#         env_content = """# Case Study Chatbot Configuration

# # Choose your LLM provider (uncomment one):

# # Option 1: Gemini (Google) - Free tier available
# GEMINI_API_KEY=your_gemini_api_key_here
# LLM_PROVIDER=gemini

# # Option 2: OpenAI - Paid service
# # OPENAI_API_KEY=your_openai_api_key_here
# # LLM_PROVIDER=openai

# # Performance settings
# EMBEDDING_BATCH_SIZE=16
# VECTOR_TOP_K=8
# GRADIO_PORT=7860
# PROFESSIONAL_MODE=true
# """
#         env_file.write_text(env_content)
#         print("✅ Created .env configuration file")
#         print("🔑 IMPORTANT: Edit .env file and add your API key!")
#         return False
    
#     # Check if API key is configured
#     try:
#         from dotenv import load_dotenv
#         load_dotenv()
        
#         gemini_key = os.getenv("GEMINI_API_KEY")
#         openai_key = os.getenv("OPENAI_API_KEY")
        
#         if (not gemini_key or gemini_key == "your_gemini_api_key_here") and \
#            (not openai_key or openai_key == "your_openai_api_key_here"):
#             print("⚠️ API key not configured!")
#             print("\n📝 Please edit .env file and add your API key")
#             print("\n🔗 Get API keys:")
#             print("   🔮 Gemini (Free): https://makersuite.google.com/app/apikey")
#             print("   🤖 OpenAI (Paid): https://platform.openai.com/api-keys")
#             print("\n💡 After adding your key, run this script again")
#             return False
        
#         print("✅ API key configured")
#         return True
        
#     except ImportError:
#         print("⚠️ python-dotenv not available, installing...")
#         subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
#         return setup_environment()  # Retry after installation

# def create_sample_data():
#     """Create sample data if needed."""
#     print("📚 Checking sample data...")
    
#     data_dir = Path("data")
#     data_dir.mkdir(exist_ok=True)
    
#     # Check if we have any data files
#     has_data = any([
#         (data_dir / "cases.docx").exists(),
#         (data_dir / "batch_1" / "advanced_cases.docx").exists(),
#         list(data_dir.glob("*.docx"))
#     ])
    
#     if not has_data:
#         print("📝 Creating sample business case studies...")
        
#         try:
#             # Try to run the sample data creator
#             if Path("create_sample_data.py").exists():
#                 subprocess.run([sys.executable, "create_sample_data.py"], check=True)
#                 print("✅ Sample data created successfully")
#             else:
#                 # Create basic sample data manually
#                 create_basic_sample_data()
#                 print("✅ Basic sample data created")
                
#         except Exception as e:
#             print(f"⚠️ Could not create full sample data: {e}")
#             create_basic_sample_data()
#             print("✅ Basic fallback data created")
#     else:
#         print("✅ Sample data available")

# def create_basic_sample_data():
#     """Create basic sample data as fallback."""
#     data_dir = Path("data")
#     data_dir.mkdir(exist_ok=True)
    
#     # Create simple text file with case studies
#     sample_cases = """BUSINESS CASE STUDIES COLLECTION

# === Case Study 1: Customer Service Transformation ===
# Problem: TechCorp faced declining customer satisfaction with 48-hour email response times and limited support coverage.
# Solution: Implemented automated ticket routing, hired additional staff, provided comprehensive training, and established 24/7 coverage.
# Result: Customer satisfaction improved from 62% to 89%, response time reduced to 4 hours, customer retention increased 23%, ROI of 340%.

# === Case Study 2: Employee Retention Initiative ===
# Problem: GlobalManufacturing had 45% annual turnover vs 18% industry average due to limited advancement, poor compensation, and work-life balance issues.
# Solution: Created career progression paths, adjusted salaries to market rates, introduced flexible work, allocated $50k for training, and established mentorship programs.
# Result: Turnover decreased to 15%, saving $2.1M in recruitment costs, satisfaction increased from 3.2 to 4.6/5, productivity up 28%.

# === Case Study 3: Digital Transformation ===
# Problem: RetailPlus traditional retail chain saw 35% decline in foot traffic, only 5% online sales, and legacy systems hindering inventory management.
# Solution: Developed omnichannel e-commerce platform with mobile app, integrated inventory system, click-and-collect services, personalized marketing, and staff training.
# Result: Online sales grew to 35% of revenue, overall sales up 22%, customer satisfaction improved to 4.3/5, $2.8M investment paid back in 14 months.

# === Case Study 4: Cost Reduction Initiative ===
# Problem: ManufacturingCorp faced rising operational costs, energy up 40%, maintenance consuming 18% of revenue.
# Solution: Implemented predictive maintenance, lean manufacturing principles, energy optimization, supplier diversification, and process automation.
# Result: Total costs reduced 28%, efficiency up 45%, quality defects down 52%, annual savings of $1.8M with $650k investment.

# === Case Study 5: Leadership Development ===
# Problem: FinanceFirst had 65% of managers retiring soon, low internal promotion rates, and declining mid-level engagement.
# Solution: 18-month leadership development program, executive coaching, job rotation, mentoring network, and competency assessments.
# Result: Internal promotions up to 78%, leadership pipeline strength improved 85%, engagement increased 32%, external hiring costs reduced $800k annually.

# === Common Questions & Answers ===
# Q: What are the most common business problems?
# A: Customer service issues, employee retention challenges, operational inefficiencies, digital transformation needs, and leadership development gaps.

# Q: How long do improvement initiatives typically take?
# A: Most show measurable results in 6-18 months, with full transformation achieved in 2-3 years depending on scope and complexity.

# Q: What factors ensure successful implementation?
# A: Strong leadership commitment, adequate resource allocation, clear communication strategies, employee training and support, and regular performance monitoring.

# Q: How do companies measure ROI on improvement initiatives?
# A: Through cost savings, revenue increases, efficiency gains, reduced turnover costs, and improved customer satisfaction leading to higher retention rates.
# """
    
#     fallback_path = data_dir / "sample_cases.txt"
#     fallback_path.write_text(sample_cases)

# def test_system_imports():
#     """Test if the system components can be imported."""
#     print("🧪 Testing system components...")
    
#     try:
#         # Test core imports
#         print("   Testing configuration...", end=" ")
#         from src.config import get_logger
#         print("✅")
        
#         print("   Testing data loader...", end=" ")
#         from src.data_loader import preprocess_text
#         print("✅")
        
#         print("   Testing embeddings...", end=" ")
#         from src.embeddings import get_embedding_manager
#         print("✅")
        
#         print("   Testing knowledge graph...", end=" ")
#         from src.knowledge_graph import get_knowledge_graph
#         print("✅")
        
#         print("   Testing retriever...", end=" ")
#         from src.retriever import get_retriever
#         print("✅")
        
#         print("   Testing agent...", end=" ")
#         from src.agent import get_agent
#         print("✅")
        
#         print("   Testing chatbot...", end=" ")
#         from src.chatbot import handle_message
#         print("✅")
        
#         print("   Testing UI...", end=" ")
#         from src.ui import EnhancedChatbotUI
#         print("✅")
        
#         print("✅ All system components ready")
#         return True
        
#     except ImportError as e:
#         print(f"❌ Import error: {e}")
#         print("💡 Try running: python comprehensive_setup.py")
#         return False
#     except Exception as e:
#         print(f"⚠️ System test error: {e}")
#         return True  # Continue anyway

# def launch_ui():
#     """Launch the enhanced UI."""
#     print("🚀 Launching enhanced chatbot UI...")
#     print("\n" + "━" * 60)
#     print("🌐 Starting web server...")
#     print("📊 Initializing AI components...")
#     print("🎨 Loading enhanced interface...")
#     print("━" * 60)
    
#     try:
#         # Import and launch the UI
#         from src.ui import EnhancedChatbotUI
        
#         ui = EnhancedChatbotUI()
#         ui.launch()
        
#     except KeyboardInterrupt:
#         print("\n👋 Chatbot stopped by user")
#     except ImportError as e:
#         print(f"❌ UI import error: {e}")
#         print("💡 Try: python comprehensive_setup.py")
#         return False
#     except Exception as e:
#         print(f"❌ UI launch error: {e}")
#         print("💡 Check the error details above and try:")
#         print("   1. python comprehensive_setup.py")
#         print("   2. python src/ui.py")
#         return False
    
#     return True

# def show_quick_help():
#     """Show quick help and usage tips."""
#     print("""
# 💡 QUICK TIPS FOR USING YOUR CHATBOT:

# 🎯 Try these example queries:
#    • "What are effective customer retention strategies?"
#    • "Analyze digital transformation success factors"
#    • "Compare cost reduction approaches"
#    • "Recommend solutions for employee engagement"

# 🎨 UI Features:
#    • Click example cards to auto-populate queries
#    • Check session stats in the right sidebar
#    • Responses include confidence scores and sources
#    • Mobile-friendly responsive design

# 🔧 Troubleshooting:
#    • If responses are slow, check your internet connection
#    • For import errors, run: python comprehensive_setup.py
#    • For API issues, verify your .env file has valid keys

# 📚 Learn More:
#    • Check the README.md for detailed documentation
#    • View logs in setup.log for debugging information
#     """)

# def main():
#     """Main launcher function."""
#     print_banner()
    
#     # Parse command line arguments
#     run_setup = "--setup" in sys.argv or "--help" in sys.argv
#     show_help = "--help" in sys.argv
#     quick_mode = "--quick" in sys.argv
    
#     if show_help:
#         print("""
# 🚀 Case Study Chatbot Launcher

# Usage: python run_chatbot.py [options]

# Options:
#   --setup     Run full setup and validation
#   --quick     Skip some checks for faster startup  
#   --help      Show this help message

# Examples:
#   python run_chatbot.py           # Normal launch
#   python run_chatbot.py --setup   # Full setup first
#   python run_chatbot.py --quick    # Quick launch
#         """)
#         show_quick_help()
#         return
    
#     start_time = time.time()
    
#     try:
#         # Step 1: Basic requirements
#         if not check_requirements():
#             return
        
#         # Step 2: Dependencies (skip in quick mode unless missing)
#         if not quick_mode or "--setup" in sys.argv:
#             install_dependencies()
        
#         # Step 3: Environment setup
#         if not setup_environment():
#             print("\n🔑 Please configure your API key in the .env file, then run again.")
#             return
        
#         # Step 4: Sample data
#         if not quick_mode:
#             create_sample_data()
        
#         # Step 5: System test (optional in quick mode)
#         if not quick_mode:
#             if not test_system_imports():
#                 print("⚠️ Some components may not work properly")
#                 user_input = input("Continue anyway? (y/n): ").lower().strip()
#                 if user_input not in ['y', 'yes']:
#                     print("💡 Try running: python comprehensive_setup.py")
#                     return
        
#         # Step 6: Launch UI
#         setup_time = time.time() - start_time
#         print(f"\n✅ Setup completed in {setup_time:.1f} seconds")
        
#         if launch_ui():
#             print("\n🎉 Chatbot session completed successfully!")
#         else:
#             print("\n❌ UI launch failed. Check error messages above.")
    
#     except KeyboardInterrupt:
#         print("\n🛑 Setup interrupted by user")
#     except Exception as e:
#         print(f"\n💥 Unexpected error: {e}")
#         print("📋 For detailed diagnosis, run: python comprehensive_setup.py")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Simple launcher for Legal Document AI Assistant.
Focused on arbitration and legal document analysis.
"""

import sys
import os
import subprocess
from pathlib import Path

def print_banner():
    """Print simple startup banner."""
    print("""
⚖️ Legal Document AI Assistant
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🧠 AI Analysis  📚 Legal Documents  ⚖️ Arbitration Cases
    """)

def check_basic_requirements():
    """Check basic system requirements."""
    print("🔍 Checking system...")
    
    # Check Python version
    if sys.version_info < (3, 10):
        print(f"❌ Python 3.10+ required, found {sys.version_info.major}.{sys.version_info.minor}")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} OK")
    
    # Check project structure
    if not Path("src").exists():
        print("❌ Please run from project directory (where src/ folder is)")
        return False
    
    print("✅ Project structure OK")
    return True

def install_required_packages():
    """Install essential packages only."""
    print("📦 Installing required packages...")
    
    essential_packages = [
        "gradio>=3.50.0",
        "python-dotenv>=1.0.0",
        "sentence-transformers>=2.2.2",
        "faiss-cpu>=1.7.4",
        "networkx>=3.1.0",
        "python-docx>=0.8.11",
        "google-generativeai>=0.3.0",
        "openai>=1.3.0"
    ]
    
    for package in essential_packages:
        try:
            package_name = package.split(">=")[0]
            if package_name == "python-dotenv":
                __import__("dotenv")
            elif package_name == "python-docx":
                __import__("docx")
            elif package_name == "faiss-cpu":
                __import__("faiss")
            elif package_name == "google-generativeai":
                __import__("google.generativeai")
            elif package_name == "sentence-transformers":
                __import__("sentence_transformers")
            else:
                __import__(package_name.replace("-", "_"))
        except ImportError:
            print(f"   Installing {package_name}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package, "--quiet"
            ])
    
    print("✅ Packages ready")

def setup_api_key():
    """Setup API key configuration."""
    print("🔧 Checking API configuration...")
    
    env_file = Path(".env")
    if not env_file.exists():
        print("📝 Creating .env configuration...")
        env_content = """# Legal Document AI Assistant Configuration

# Choose your AI provider:

# Option 1: Gemini (Google) - Free tier available
GEMINI_API_KEY=your_gemini_api_key_here
LLM_PROVIDER=gemini

# Option 2: OpenAI - Paid service
# OPENAI_API_KEY=your_openai_api_key_here
# LLM_PROVIDER=openai

# System settings
GRADIO_PORT=7860
"""
        env_file.write_text(env_content)
        print("✅ Created .env file")
        print("\n🔑 IMPORTANT: Edit .env file and add your API key!")
        print("   🔮 Gemini (Free): https://makersuite.google.com/app/apikey")
        print("   🤖 OpenAI (Paid): https://platform.openai.com/api-keys")
        return False
    
    # Check if API key is configured
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if (not gemini_key or gemini_key == "your_gemini_api_key_here") and \
           (not openai_key or openai_key == "your_openai_api_key_here"):
            print("⚠️ API key not configured!")
            print("📝 Please edit .env file and add your API key, then run again")
            return False
        
        print("✅ API key configured")
        return True
        
    except ImportError:
        install_required_packages()
        return setup_api_key()

def check_documents():
    """Check for legal documents."""
    print("📚 Checking for documents...")
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Look for any .docx files
    import glob
    doc_files = glob.glob(str(data_dir / "*.docx")) + glob.glob(str(data_dir / "*" / "*.docx"))
    
    if doc_files:
        print(f"✅ Found {len(doc_files)} document(s)")
        for doc_file in doc_files[:3]:  # Show first 3
            print(f"   📄 {Path(doc_file).name}")
        if len(doc_files) > 3:
            print(f"   ... and {len(doc_files) - 3} more")
    else:
        print("📝 No documents found - will create sample legal data")
        create_sample_legal_data()

def create_sample_legal_data():
    """Create sample legal/arbitration data."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    sample_content = """ARBITRATION AND LEGAL CASES COLLECTION

=== Arbitration Timeline and Procedures ===
The standard arbitration process typically takes 6-18 months from initiation to final award. Emergency arbitration procedures can be completed within 14-30 days for urgent matters requiring immediate relief.

Key procedural steps include:
1. Filing of Request for Arbitration (30 days after dispute arises)
2. Response to Request (30 days from notification)
3. Constitution of Arbitral Tribunal (60 days)
4. Preliminary Hearing (within 90 days)
5. Evidence Phase (3-6 months)
6. Final Hearing (2-4 weeks)
7. Award Issuance (30-60 days post-hearing)

=== Arbitration Costs and Claims Analysis ===
Arbitration costs typically range from $50,000 to $500,000 depending on claim value and complexity. Cost breakdown includes:
- Administrative fees: 5-15% of total costs
- Arbitrator fees: 60-70% of total costs  
- Legal representation: 20-30% of total costs
- Other expenses: 5-10% of total costs

Claims under $1M average $75,000 in total arbitration costs.
Claims $1M-$10M average $250,000 in total arbitration costs.
Claims over $10M can exceed $500,000 in total arbitration costs.

=== Validity of Arbitration Clauses ===
For arbitration clauses to be valid and enforceable:
1. Clear and unambiguous language required
2. Mutual agreement by all parties
3. Proper scope definition (what disputes are covered)
4. Designated arbitration rules and institution
5. Seat/venue of arbitration specified
6. Governing law identified
7. Number and appointment of arbitrators defined

Invalid clauses often result from:
- Vague or contradictory language
- Unconscionable terms
- Lack of mutuality
- Procedural impossibilities

=== Emergency Arbitration Conditions ===
Emergency arbitration is available when:
1. Immediate harm would occur without relief
2. Regular arbitration process would be too slow
3. Interim measures are necessary to preserve status quo
4. Irreparable damage is imminent
5. Balance of harm favors emergency relief

Emergency arbitrator powers include:
- Temporary restraining orders
- Asset preservation orders
- Preliminary injunctions
- Security for costs orders

=== Decision Trees and Award Enforcement ===
Arbitral awards are generally final and binding, with limited grounds for challenge:
- Procedural irregularities
- Arbitrator misconduct or bias
- Award exceeds scope of arbitration agreement
- Award violates public policy
- Lack of proper notice or opportunity to be heard

Enforcement success rates:
- Domestic awards: 95% enforcement rate
- International awards: 85% enforcement rate
- Challenge success rate: Less than 10%

=== Frequently Asked Questions ===
Q: How long does arbitration take?
A: Standard arbitration takes 6-18 months; emergency arbitration takes 14-30 days.

Q: What are typical arbitration costs?
A: Costs range from $50,000 to $500,000 depending on claim size and complexity.

Q: Can arbitration awards be appealed?
A: Appeals are very limited, with less than 10% success rate for challenges.

Q: What makes an arbitration clause valid?
A: Clear language, mutual agreement, proper scope, designated rules, and specified procedures.

Q: When is emergency arbitration available?
A: When immediate harm would occur and regular arbitration would be too slow to prevent irreparable damage.
"""
    
    sample_file = data_dir / "legal_arbitration_cases.txt"
    sample_file.write_text(sample_content)
    print(f"✅ Created sample legal data: {sample_file}")

def launch_legal_assistant():
    """Launch the legal document assistant."""
    print("🚀 Starting Legal Document AI Assistant...")
    
    try:
        from src.ui import SimpleChatbotUI
        ui = SimpleChatbotUI()
        ui.launch()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all files are in place and try again")
        return False
    except Exception as e:
        print(f"❌ Launch error: {e}")
        return False
    
    return True

def main():
    """Main launcher function."""
    print_banner()
    
    try:
        # Step 1: Basic checks
        if not check_basic_requirements():
            return
        
        # Step 2: Install packages
        install_required_packages()
        
        # Step 3: Setup API key
        if not setup_api_key():
            return
        
        # Step 4: Check documents
        check_documents()
        
        # Step 5: Launch
        print("\n🚀 Launching Legal Document AI Assistant...")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        if launch_legal_assistant():
            print("\n🎉 Session completed successfully!")
        else:
            print("\n❌ Failed to launch assistant")
    
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
    except Exception as e:
        print(f"\n💥 Error: {e}")

if __name__ == "__main__":
    main()