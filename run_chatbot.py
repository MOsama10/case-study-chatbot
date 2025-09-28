
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
