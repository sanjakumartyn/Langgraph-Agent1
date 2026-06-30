import asyncio
from rag_app.pdf_generator import generate_company_proposal_pdf
from pathlib import Path

async def main():
    company = "Amazon"
    FRONTEND_DIR = Path("frontend")
    filename = f"Proposal_{company.replace(' ', '_')}_Q3.pdf"
    file_path = FRONTEND_DIR / filename
    
    print(f"Generating PDF for {company}...")
    await asyncio.to_thread(generate_company_proposal_pdf, company, str(file_path))
    print(f"PDF generated successfully at: {file_path.absolute()}")

if __name__ == "__main__":
    asyncio.run(main())
