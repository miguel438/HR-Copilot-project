"""Generate additional synthetic CV PDFs for training-set augmentation only.

Output goes to training/synthetic_resumes/, never to data/resumes/ - that separation is the whole
point. data/resumes/ is what ingest.py loads into the production Qdrant collection the live app
(and the n8n flow that calls it) actually searches; these PDFs are read by
build_pairs_synthetic.py into a separate, training-only collection and never reach the running
system. See training/README.md for the full augmentation pipeline this feeds.

Each candidate is written as a deliberately unambiguous fit (or non-fit) for exactly one job
family in training/requisitions.json, chosen to fill gaps the original 10-CV corpus had:
req-t11 (maritime) and req-t14 (fullstack) previously had zero genuinely matching candidates, and
the class balance was heavily skewed toward no-fit (111/13/16). A few are given a years_experience
just under a requisition's minimum on purpose, mirroring the years_shortfall pattern
Noah Kim/Maya Levi already established, rather than leaving that feature under-exercised.

Layout matches src/ingest.py's parse_resume() exactly: name on line 1, role title on line 2, an
"Email:" line, a "Skills:" line (comma-separated), and a "Years of experience:" line - the only
lines that function parses out as structured metadata. Everything else is free text the Evaluator
LLM reads directly, same as the original 10 CVs.
"""

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

OUT_DIR = Path(__file__).resolve().parent / "synthetic_resumes"

CANDIDATES = [
    {
        "name": "Marcus Webb",
        "role": "Backend Engineer",
        "email": "marcus.webb@example.com",
        "summary": [
            "Backend engineer with 4 years of experience building Python web services and APIs",
            "for early-stage and growth-stage startups, deployed on AWS.",
        ],
        "skills": ["Python", "Flask", "REST APIs", "AWS (EC2, S3, RDS)", "PostgreSQL", "Docker", "Git"],
        "experience": [
            "- Backend Engineer, Fernway Logistics (2022-present): Built the Flask-based order-routing "
            "API and its PostgreSQL data model; deployed and maintained the service on AWS EC2 and RDS.",
            "- Software Engineer, BrightCart (2021-2022): Contributed REST endpoints and Docker-based "
            "local dev tooling for a small e-commerce backend team.",
        ],
        "education": "B.Sc. Computer Science, University of Toronto (2021)",
        "years": 4,
    },
    {
        "name": "Priya Nair",
        "role": "Senior Frontend Engineer",
        "email": "priya.nair@example.com",
        "summary": [
            "Frontend engineer with 6 years of experience building React/TypeScript applications, "
            "including component architecture, state management and design-system adoption.",
        ],
        "skills": ["React", "TypeScript", "Next.js", "GraphQL", "CSS/Tailwind", "Jest", "Webpack"],
        "experience": [
            "- Senior Frontend Engineer, Ledgerly (2020-present): Led the migration of the main "
            "dashboard from a legacy jQuery codebase to React/TypeScript with Next.js and GraphQL.",
            "- Frontend Engineer, Kinetiq Media (2018-2020): Built React components for a "
            "subscriber-facing video platform, integrating with REST and GraphQL backends.",
        ],
        "education": "B.Sc. Computer Science, University of Waterloo (2018)",
        "years": 6,
    },
    {
        "name": "Chloe Bennett",
        "role": "Senior Data Analyst",
        "email": "chloe.bennett@example.com",
        "summary": [
            "Data analyst with 5 years of experience building SQL data models and dashboards for "
            "revenue and operations teams, with hands-on Python for deeper analysis.",
        ],
        "skills": ["SQL", "Python (pandas)", "dbt", "Snowflake", "Looker", "A/B testing", "Excel"],
        "experience": [
            "- Senior Data Analyst, Northlake Foods (2021-present): Owns the dbt/Snowflake data models "
            "powering the company's Looker dashboards for revenue and supply-chain reporting.",
            "- Data Analyst, Vantage Retail (2019-2021): Wrote SQL reporting and ran A/B tests on "
            "pricing experiments, presenting findings to the merchandising team.",
        ],
        "education": "B.Sc. Statistics, McGill University (2019)",
        "years": 5,
    },
    {
        "name": "Tom Richardson",
        "role": "DevOps Engineer",
        "email": "tom.richardson@example.com",
        "summary": [
            "DevOps engineer with 4 years of experience running Kubernetes clusters and AWS "
            "infrastructure, and building the CI/CD pipelines that deploy onto them.",
        ],
        "skills": ["AWS (EKS, EC2, IAM)", "Kubernetes", "Terraform", "Docker", "CI/CD (GitHub Actions)", "Python (scripting)"],
        "experience": [
            "- DevOps Engineer, Faircove Health (2022-present): Runs the company's AWS EKS cluster and "
            "wrote the Terraform modules provisioning its networking and IAM roles.",
            "- Infrastructure Engineer, Loop Robotics (2021-2022): Built GitHub Actions CI/CD pipelines "
            "and Dockerized the deployment process for three internal services.",
        ],
        "education": "B.Sc. Computer Engineering, University of British Columbia (2021)",
        "years": 4,
    },
    {
        "name": "Grace Kim",
        "role": "Senior Product Manager",
        "email": "grace.kim@example.com",
        "summary": [
            "Product manager with 7 years of experience leading consumer mobile products from "
            "discovery through launch, partnering closely with engineering and design.",
        ],
        "skills": ["product strategy", "roadmapping", "user research", "SQL (for analysis)", "A/B testing", "Jira", "mobile/consumer app domain knowledge"],
        "experience": [
            "- Senior Product Manager, Trailhead Fitness App (2020-present): Owns the onboarding and "
            "retention roadmap; ran the A/B test program that lifted 30-day retention by 15%.",
            "- Product Manager, Nestwell Home (2017-2020): Led the mobile app's redesign from user "
            "research through launch, working directly with engineering and design.",
        ],
        "education": "MBA, INSEAD (2017); B.A. Economics, University of Chicago (2013)",
        "years": 7,
    },
    {
        "name": "Lucas Ferreira",
        "role": "UX Designer",
        "email": "lucas.ferreira@example.com",
        "summary": [
            "UX designer with 3 years of experience designing web interfaces for early-stage "
            "products, from user research through high-fidelity prototypes.",
        ],
        "skills": ["Figma", "wireframing", "prototyping", "user research", "usability testing"],
        "experience": [
            "- UX Designer, Harborlight Insurance (2022-present): Designed the claims-filing flow in "
            "Figma and ran usability testing sessions that cut task-completion time by 20%.",
            "- Junior Designer, Studio Marrow (2021-2022): Built wireframes and prototypes for client "
            "marketing sites.",
        ],
        "education": "B.Des. Interaction Design, OCAD University (2021)",
        "years": 3,
    },
    {
        "name": "Isabella Rossi",
        "role": "Account Executive",
        "email": "isabella.rossi@example.com",
        "summary": [
            "B2B SaaS sales professional with 6 years of experience running full-cycle sales for "
            "mid-market and enterprise accounts.",
        ],
        "skills": ["Salesforce", "outbound prospecting", "solution selling", "negotiation", "quota attainment", "HubSpot"],
        "experience": [
            "- Account Executive, Ferrovia Analytics (2021-present): Closed $2.1M ARR in FY2025 "
            "selling analytics software to mid-market and enterprise accounts; consistently exceeded "
            "quota.",
            "- Account Executive, Bramwell CRM (2019-2021): Ran full-cycle sales from outbound "
            "prospecting through close for mid-market customers.",
        ],
        "education": "B.A. Business, Queen's University (2019)",
        "years": 6,
    },
    {
        "name": "Ben Carter",
        "role": "Full-Stack Engineer",
        "email": "ben.carter@example.com",
        "summary": [
            "Full-stack engineer with 4 years of experience building React/TypeScript front ends "
            "backed by Python/Django services on relational databases.",
        ],
        "skills": ["React", "TypeScript", "Python", "Django", "REST APIs", "PostgreSQL", "Docker"],
        "experience": [
            "- Full-Stack Engineer, Meadowlark Bookings (2022-present): Builds both the React/"
            "TypeScript booking UI and the Django/PostgreSQL REST API behind it, deployed via Docker.",
            "- Full-Stack Developer, Cobblestone Studio (2021-2022): Built small business websites with "
            "React front ends and Python/Flask backends.",
        ],
        "education": "B.Sc. Computer Science, Simon Fraser University (2021)",
        "years": 4,
    },
    {
        "name": "Hannah Osei",
        "role": "Software Engineer",
        "email": "hannah.osei@example.com",
        "summary": [
            "Software engineer with 3 years of experience building REST APIs in Python against "
            "PostgreSQL, deployed with Docker.",
        ],
        "skills": ["Python", "Docker", "PostgreSQL", "REST APIs", "Git", "Linux"],
        "experience": [
            "- Software Engineer, Ridgeline Analytics (2022-present): Builds and maintains Python REST "
            "APIs against PostgreSQL, packaged with Docker for deployment.",
            "- Junior Software Engineer, Pinegrove Systems (2021-2022): Wrote internal Python tooling "
            "and REST integrations against the company's PostgreSQL database.",
        ],
        "education": "B.Sc. Computer Science, University of Alberta (2021)",
        "years": 3,
    },
    {
        "name": "Derek Holloway",
        "role": "Licensed Ship Captain",
        "email": "derek.holloway@example.com",
        "summary": [
            "Licensed ship captain with 9 years of experience commanding freight vessels, managing "
            "crew schedules and maintaining port compliance across international routes.",
        ],
        "skills": ["maritime navigation", "freight logistics", "crew management", "port compliance", "vessel safety", "radar/ECDIS systems"],
        "experience": [
            "- Ship Captain, Northstar Freight Lines (2019-present): Commands a freight vessel on "
            "transatlantic routes, owning crew scheduling, navigation, and port compliance "
            "documentation.",
            "- First Officer, Harborline Shipping (2016-2019): Managed watch schedules and supported "
            "freight logistics planning under the captain's command.",
        ],
        "education": "Master Mariner License, Maritime Institute of Technology (2016)",
        "years": 9,
    },
    {
        "name": "Rachel Adler",
        "role": "Registered Nurse",
        "email": "rachel.adler@example.com",
        "summary": [
            "Registered nurse with 5 years of experience in outpatient clinical care, patient "
            "documentation and electronic health records.",
        ],
        "skills": ["patient care", "electronic health records (EHR)", "clinical documentation", "BLS/ACLS certified", "medication administration"],
        "experience": [
            "- Registered Nurse, Willowbrook Outpatient Clinic (2021-present): Provides direct patient "
            "care and maintains clinical documentation in the clinic's electronic health record "
            "system.",
            "- Registered Nurse, Cedarview Medical Group (2019-2021): Administered medications and "
            "supported physicians during outpatient visits.",
        ],
        "education": "B.Sc. Nursing, University of Ottawa (2019)",
        "years": 5,
    },
]


def render_resume(candidate: dict, path: Path) -> None:
    doc = canvas.Canvas(str(path), pagesize=letter)
    doc.setFont("Helvetica", 10)
    width, height = letter
    x, y = 72, height - 72
    line_height = 14

    def line(text: str = "") -> None:
        nonlocal y
        doc.drawString(x, y, text)
        y -= line_height

    line(candidate["name"])
    line(candidate["role"])
    line(f"Email: {candidate['email']}")
    line()
    line("Summary:")
    for part in candidate["summary"]:
        line(part)
    line()
    line("Skills: " + ", ".join(candidate["skills"]))
    line()
    line("Experience:")
    for entry in candidate["experience"]:
        line(entry)
    line()
    line(f"Education: {candidate['education']}")
    line()
    line(f"Years of experience: {candidate['years']}")

    doc.save()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for candidate in CANDIDATES:
        slug = candidate["name"].lower().replace(" ", "_")
        path = OUT_DIR / f"{slug}.pdf"
        render_resume(candidate, path)
        print(f"  wrote {path.relative_to(OUT_DIR.parent.parent)}")
    print(f"\n{len(CANDIDATES)} synthetic resumes written to {OUT_DIR}")


if __name__ == "__main__":
    main()
