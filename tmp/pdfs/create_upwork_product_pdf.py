from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
TMP_DIR = ROOT / "tmp" / "pdfs"
OUT = ROOT / "output" / "pdf" / "agentic-ai-workflow-studio-upwork-product-document.pdf"

PAGE_W, PAGE_H = letter
MARGIN = 40
CONTENT_W = PAGE_W - (2 * MARGIN)

BG = colors.HexColor("#0d1621")
SURFACE = colors.HexColor("#152232")
SURFACE_2 = colors.HexColor("#1d3045")
SURFACE_3 = colors.HexColor("#223a55")
TEXT = colors.HexColor("#f5f7fb")
MUTED = colors.HexColor("#b7c3d6")
LINE = colors.HexColor("#355069")
ACCENT = colors.HexColor("#57c6f2")
ACCENT_2 = colors.HexColor("#71e2b0")
ACCENT_3 = colors.HexColor("#f3c95a")
DANGER = colors.HexColor("#f08383")

SCREENSHOTS = {
    "home": TMP_DIR / "demo-home-populated-dark.png",
    "compose": TMP_DIR / "demo-compose-populated-dark.png",
    "chat": TMP_DIR / "demo-chat-populated-dark.png",
    "studio": TMP_DIR / "demo-studio-populated-dark.png",
    "workflows": TMP_DIR / "demo-workflows-populated-dark.png",
    "memory": TMP_DIR / "demo-memory-populated-dark.png",
    "rag": TMP_DIR / "demo-rag-populated-dark.png",
}


def wrap_text(c: canvas.Canvas, text: str, width: float, font: str, size: int) -> list[str]:
    paragraphs = text.split("\n")
    lines: list[str] = []
    for paragraph in paragraphs:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if c.stringWidth(candidate, font, size) <= width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def draw_paragraph(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    *,
    font: str = "Helvetica",
    size: int = 10,
    leading: int = 14,
    color=TEXT,
) -> float:
    c.setFillColor(color)
    c.setFont(font, size)
    for line in wrap_text(c, text, width, font, size):
        c.drawString(x, y, line)
        y -= leading
    return y


def draw_label(c: canvas.Canvas, text: str, x: float, y: float, color=ACCENT) -> None:
    c.setFillColor(color)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x, y, text.upper())


def draw_title(c: canvas.Canvas, title: str, subtitle: str, *, page_label: str) -> float:
    draw_label(c, page_label, MARGIN, PAGE_H - 38)
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 26)
    y = PAGE_H - 68
    for line in wrap_text(c, title, CONTENT_W, "Helvetica-Bold", 26):
        c.drawString(MARGIN, y, line)
        y -= 30
    y -= 2
    return draw_paragraph(c, subtitle, MARGIN, y, CONTENT_W, size=11, leading=16, color=MUTED)


def footer(c: canvas.Canvas, page_number: int) -> None:
    c.setStrokeColor(LINE)
    c.line(MARGIN, 28, PAGE_W - MARGIN, 28)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8)
    c.drawString(MARGIN, 16, "Agentic Workflow Studio product document")
    c.drawRightString(PAGE_W - MARGIN, 16, f"Page {page_number}")


def new_page(c: canvas.Canvas, page_number: int) -> None:
    c.setFillColor(BG)
    c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)
    footer(c, page_number)


def pill(c: canvas.Canvas, x: float, y: float, text: str, *, stroke=ACCENT, fill=SURFACE) -> float:
    width = c.stringWidth(text, "Helvetica-Bold", 8) + 20
    c.setStrokeColor(stroke)
    c.setFillColor(fill)
    c.roundRect(x, y - 5, width, 20, 8, stroke=1, fill=1)
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 8)
    c.drawString(x + 10, y + 1, text)
    return x + width + 8


def panel(c: canvas.Canvas, x: float, y_top: float, w: float, h: float, *, fill=SURFACE, stroke=LINE) -> None:
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.roundRect(x, y_top - h, w, h, 10, stroke=1, fill=1)


def info_card(
    c: canvas.Canvas,
    x: float,
    y_top: float,
    w: float,
    h: float,
    title: str,
    body: str,
    *,
    accent=ACCENT,
    fill=SURFACE,
) -> None:
    panel(c, x, y_top, w, h, fill=fill)
    c.setFillColor(accent)
    c.rect(x + 14, y_top - 18, 26, 4, stroke=0, fill=1)
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x + 14, y_top - 34, title)
    draw_paragraph(c, body, x + 14, y_top - 52, w - 28, size=9, leading=12, color=MUTED)


def bullet_list(
    c: canvas.Canvas,
    items: Iterable[tuple[str, str]],
    x: float,
    y: float,
    width: float,
    *,
    bullet_color=ACCENT,
    gap: int = 10,
    title_size: int = 10,
    body_size: int = 9,
    body_leading: int = 13,
) -> float:
    for heading, body in items:
        c.setFillColor(bullet_color)
        c.circle(x + 4, y - 4, 3, stroke=0, fill=1)
        c.setFillColor(TEXT)
        c.setFont("Helvetica-Bold", title_size)
        c.drawString(x + 16, y - 8, heading)
        y = draw_paragraph(c, body, x + 16, y - 24, width - 16, size=body_size, leading=body_leading, color=MUTED)
        y -= gap
    return y


def screenshot_block(
    c: canvas.Canvas,
    path: Path,
    x: float,
    y_top: float,
    w: float,
    h: float,
    *,
    title: str,
    subtitle: str,
) -> None:
    panel(c, x, y_top, w, h + 42, fill=SURFACE_2)
    image = Image.open(path).convert("RGB")
    img_w, img_h = image.size
    image_box_h = h - 34
    scale = min(w / img_w, image_box_h / img_h)
    draw_w = img_w * scale
    draw_h = img_h * scale
    draw_x = x + (w - draw_w) / 2
    draw_y = y_top - 14 - draw_h
    c.drawImage(ImageReader(image), draw_x, draw_y, width=draw_w, height=draw_h, preserveAspectRatio=True, mask="auto")
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x + 12, y_top - h - 8, title)
    draw_paragraph(c, subtitle, x + 12, y_top - h - 20, w - 24, size=7, leading=8, color=MUTED)


def section_band(c: canvas.Canvas, title: str, body: str, *, x: float, y_top: float, w: float, h: float) -> None:
    panel(c, x, y_top, w, h, fill=SURFACE_3)
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x + 16, y_top - 24, title)
    draw_paragraph(c, body, x + 16, y_top - 44, w - 32, size=10, leading=14, color=MUTED)


def stat_card(c: canvas.Canvas, x: float, y_top: float, w: float, h: float, value: str, label: str, body: str, accent=ACCENT) -> None:
    panel(c, x, y_top, w, h, fill=SURFACE)
    c.setFillColor(accent)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(x + 14, y_top - 30, value)
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x + 14, y_top - 48, label)
    draw_paragraph(c, body, x + 14, y_top - 64, w - 28, size=8, leading=11, color=MUTED)


def draw_flow(c: canvas.Canvas, steps: list[tuple[str, str]], *, x: float, y_top: float, w: float) -> None:
    count = len(steps)
    gap = 12
    box_w = (w - (gap * (count - 1))) / count
    box_h = 90
    for index, (title, body) in enumerate(steps):
        box_x = x + index * (box_w + gap)
        fill = SURFACE if index % 2 == 0 else SURFACE_2
        info_card(c, box_x, y_top, box_w, box_h, title, body, accent=ACCENT if index % 2 == 0 else ACCENT_2, fill=fill)
        if index < count - 1:
            start_x = box_x + box_w + 4
            end_x = box_x + box_w + gap - 4
            y = y_top - 38
            c.setStrokeColor(LINE)
            c.setLineWidth(2)
            c.line(start_x, y, end_x, y)
            c.line(end_x - 6, y + 4, end_x, y)
            c.line(end_x - 6, y - 4, end_x, y)


def draw_comparison_row(
    c: canvas.Canvas,
    x: float,
    y_top: float,
    w: float,
    row_h: float,
    label: str,
    left_text: str,
    right_text: str,
    *,
    first_row: bool = False,
) -> None:
    c.setStrokeColor(LINE)
    if first_row:
        c.line(x, y_top, x + w, y_top)
    c.line(x, y_top - row_h, x + w, y_top - row_h)
    left_col = x + 170
    right_col = x + (w + 170) / 2
    c.line(left_col, y_top, left_col, y_top - row_h)
    c.line(right_col, y_top, right_col, y_top - row_h)
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x + 10, y_top - 18, label)
    draw_paragraph(c, left_text, left_col + 10, y_top - 18, right_col - left_col - 20, size=8, leading=11, color=MUTED)
    draw_paragraph(c, right_text, right_col + 10, y_top - 18, x + w - right_col - 20, size=8, leading=11, color=MUTED)


def build() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=letter)

    # Page 1
    new_page(c, 1)
    y = draw_title(
        c,
        "Custom Agentic AI Workflow Automation Platform",
        "Design, launch, monitor, and improve multi-step AI workflows with a production-oriented planner-executor foundation, visual workflow builder, knowledge retrieval, reusable memory, and deployment-ready backend services.",
        page_label="Product Overview",
    )
    y -= 6
    x = MARGIN
    for label in [
        "Workflow Builder",
        "Chat-to-Workflow",
        "Adaptive Planning",
        "Knowledge Base",
        "User Context Memory",
        "Run Monitoring",
    ]:
        x = pill(c, x, y, label)
    screenshot_block(
        c,
        SCREENSHOTS["studio"],
        MARGIN,
        y - 26,
        CONTENT_W,
        278,
        title="Workflow Builder",
        subtitle="Dark-mode builder for business process design, validation, runtime settings, and workflow request contracts.",
    )
    card_y = 184
    info_card(
        c,
        MARGIN,
        card_y,
        164,
        100,
        "What clients buy",
        "A configurable product base for internal copilots, AI workflow automation, and multi-step task execution instead of a thin single-prompt demo.",
        accent=ACCENT,
    )
    info_card(
        c,
        MARGIN + 178,
        card_y,
        164,
        100,
        "What it includes",
        "Frontend surfaces, API control plane, planner, workers, typed execution contracts, reusable capabilities, memory, triggers, feedback, and artifact delivery.",
        accent=ACCENT_2,
    )
    info_card(
        c,
        MARGIN + 356,
        card_y,
        164,
        100,
        "Why it sells",
        "It speaks to the real client need: orchestration, context, governance, and visibility across AI work that must survive beyond a chatbot trial.",
        accent=ACCENT_3,
    )
    c.showPage()

    # Page 2
    new_page(c, 2)
    y = draw_title(
        c,
        "Why buyers choose this over a basic chatbot build",
        "Most business use cases need repeatable execution, tool routing, grounded context, approval visibility, and run history. This platform packages those concerns into one implementation path.",
        page_label="Buyer Positioning",
    )
    left_x = MARGIN
    right_x = PAGE_W / 2 + 8
    info_card(
        c,
        left_x,
        y - 10,
        250,
        98,
        "From request to execution",
        "Users can submit work through structured forms, chat, or prebuilt workflows. The request then moves through a consistent job and run model.",
        accent=ACCENT,
    )
    info_card(
        c,
        right_x,
        y - 10,
        250,
        98,
        "Built for business processes",
        "Ideal for onboarding, research pipelines, document generation, internal operations assistants, support routing, and multi-tool task automation.",
        accent=ACCENT_2,
    )
    info_card(
        c,
        left_x,
        y - 126,
        250,
        98,
        "Grounded and traceable",
        "Knowledge retrieval, memory, typed step inputs, and run telemetry help teams understand why the system acted and what it produced.",
        accent=ACCENT_3,
    )
    info_card(
        c,
        right_x,
        y - 126,
        250,
        98,
        "Ready to customize",
        "The product is designed for client-specific capabilities, data sources, policies, and deployment choices without rebuilding the stack from scratch.",
        accent=ACCENT,
    )
    section_band(
        c,
        "Ideal buyer profile",
        "Product owners, operations leads, innovation teams, and technical founders who want AI to complete structured work across tools and documents rather than respond with standalone text.",
        x=MARGIN,
        y_top=350,
        w=CONTENT_W,
        h=66,
    )
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(MARGIN, 250, "Business outcomes this product supports")
    y = bullet_list(
        c,
        [
            ("Faster implementation", "Start from a working product surface and runtime instead of funding a greenfield orchestration project."),
            ("Higher stakeholder confidence", "Show workflows, run state, plan revisions, and execution history in a client-friendly interface."),
            ("Cleaner handoff", "Deliver code, configuration, screenshots, and runnable workflows instead of a fragile prompt bundle."),
            ("Better long-term economics", "Extend with new capabilities, retrieval sources, or policies without replacing the original foundation."),
        ],
        MARGIN,
        232,
        CONTENT_W,
    )
    c.showPage()

    # Page 3
    new_page(c, 3)
    y = draw_title(
        c,
        "Product tour: how a user enters and launches work",
        "The UI is structured for product owners and operators who need several ways to initiate work while keeping context, inputs, and workflow controls visible.",
        page_label="Product Tour I",
    )
    screenshot_block(
        c,
        SCREENSHOTS["home"],
        MARGIN,
        y - 12,
        250,
        176,
        title="AI Workflow Workspace",
        subtitle="Entry point for request intake, workflow chat, builder access, context memory, and knowledge base.",
    )
    screenshot_block(
        c,
        SCREENSHOTS["compose"],
        PAGE_W / 2 + 8,
        y - 12,
        250,
        176,
        title="Workflow Request Builder",
        subtitle="Structured job creation with goal capture, context fields, intent review, and submission-ready inputs.",
    )
    screenshot_block(
        c,
        SCREENSHOTS["chat"],
        MARGIN,
        332,
        CONTENT_W,
        128,
        title="Chat-to-Workflow Assistant",
        subtitle="Natural-language intake that can clarify the request and route it into an executable workflow-backed job.",
    )
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(MARGIN, 156, "Client-facing value on these screens")
    bullet_list(
        c,
        [
            ("Multiple intake paths", "Teams can launch work through structured request forms, chat-assisted submission, or saved workflows."),
            ("Context-aware requests", "User identity and reusable fields reduce repeated setup across recurring work."),
            ("Safer execution handoff", "Chat stays an intake surface while workflow rules and downstream execution controls remain intact."),
        ],
        MARGIN,
        138,
        CONTENT_W,
        gap=5,
        title_size=9,
        body_size=8,
        body_leading=11,
    )
    c.showPage()

    # Page 4
    new_page(c, 4)
    y = draw_title(
        c,
        "Product tour: designing, versioning, and running workflows",
        "The product supports explicit workflow authoring and lifecycle management so buyers can move from prototype flows to repeatable automation.",
        page_label="Product Tour II",
    )
    screenshot_block(
        c,
        SCREENSHOTS["studio"],
        MARGIN,
        y - 8,
        308,
        208,
        title="Workflow Builder",
        subtitle="Visual process design with step palette, flow graph, readiness checks, request contract, and runtime settings.",
    )
    screenshot_block(
        c,
        SCREENSHOTS["workflows"],
        PAGE_W / 2 + 16,
        y - 8,
        212,
        208,
        title="Saved Workflows",
        subtitle="Definitions, versions, triggers, and run history for reusable delivery assets.",
    )
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(MARGIN, 250, "Core builder and automation capabilities")
    bullet_list(
        c,
        [
            ("Visual process mapping", "Model steps, decisions, tools, inputs, outputs, and control flow before launching the automation."),
            ("Readiness and validation", "Use compile previews and workflow request contracts to catch missing inputs and setup gaps before runtime."),
            ("Versioned delivery and runtime options", "Promote draft workflows into reusable assets with triggers, run records, execution mode, and replan policy support."),
        ],
        MARGIN,
        232,
        CONTENT_W,
        gap=8,
    )
    section_band(
        c,
        "What this means for prospective clients",
        "Clients are not buying just a prompt flow. They are buying a product surface that lets their team inspect, rerun, improve, and govern workflows after the initial delivery.",
        x=MARGIN,
        y_top=110,
        w=CONTENT_W,
        h=58,
    )
    c.showPage()

    # Page 5
    new_page(c, 5)
    y = draw_title(
        c,
        "Knowledge, memory, and operational confidence",
        "The product includes dedicated screens for context retrieval and reusable user memory so workflows can be grounded, personalized, and easier to operate over time.",
        page_label="Product Tour III",
    )
    screenshot_block(
        c,
        SCREENSHOTS["rag"],
        MARGIN,
        y - 8,
        255,
        176,
        title="Knowledge Base",
        subtitle="Index and inspect documents so workflows can retrieve grounded context from workspace and knowledge collections.",
    )
    screenshot_block(
        c,
        SCREENSHOTS["memory"],
        PAGE_W / 2 + 10,
        y - 8,
        255,
        176,
        title="User Context Memory",
        subtitle="Store reusable user and project preferences so recurring work does not require repeated setup.",
    )
    stat_card(c, MARGIN, 256, 122, 88, "RAG", "Grounded answers", "Search indexed documents and attach the right context to workflow steps.", accent=ACCENT)
    stat_card(c, MARGIN + 134, 256, 122, 88, "Memory", "Reusable context", "Persist user and project details that should carry across requests.", accent=ACCENT_2)
    stat_card(c, MARGIN + 268, 256, 122, 88, "Runs", "Observable execution", "Trace workflow state, step outputs, and revisions for operational review.", accent=ACCENT_3)
    stat_card(c, MARGIN + 402, 256, 122, 88, "Feedback", "Improvement loop", "Collect structured response and outcome feedback to improve future delivery.", accent=DANGER)
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(MARGIN, 136, "Why buyers care")
    bullet_list(
        c,
        [
            ("Less prompt stuffing", "Context retrieval reduces the need to force large reference payloads into every request."),
            ("Better repeatability", "Saved preferences help teams run the same workflow pattern with less re-entry and less drift."),
            ("Operational visibility", "Run steps, attempts, execution requests, and feedback surfaces help buyers trust production use."),
        ],
        MARGIN,
        118,
        CONTENT_W,
        gap=5,
        title_size=9,
        body_size=8,
        body_leading=11,
    )
    c.showPage()

    # Page 6
    new_page(c, 6)
    y = draw_title(
        c,
        "Technical foundation clients can grow with",
        "The architecture is designed for real implementation work: a frontend product surface, API control plane, planner and worker services, typed contracts, storage, streaming events, and flexible LLM provider configuration.",
        page_label="Architecture",
    )
    draw_flow(
        c,
        [
            ("UI surfaces", "Home, Request Builder, Workflow Chat, Builder, Memory, and Knowledge Base."),
            ("API control plane", "REST and SSE APIs for jobs, plans, workflow versions, runs, downloads, and feedback."),
            ("Planner and workers", "Typed plan creation and step execution through capability and tool runtimes."),
            ("Storage and events", "Postgres for durable state and Redis Streams for orchestration events."),
        ],
        x=MARGIN,
        y_top=y - 10,
        w=CONTENT_W,
    )
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(MARGIN, 340, "Client-relevant technical capabilities")
    bullet_list(
        c,
        [
            ("Flexible model providers", "Provider settings support OpenAI, Gemini, and OpenAI-compatible endpoints so the stack can align with quality, budget, or hosting requirements."),
            ("Capability governance", "Tool and capability controls support allowlists, deny rules, contract validation, and policy-driven behavior by service or tenant."),
            ("Hybrid execution paths", "The platform supports planner-led jobs, direct chat execution for safe read-only paths, and manually authored workflows that run without planner involvement."),
            ("Deployment options", "The repository includes local Docker Compose setup and Kubernetes-oriented configuration for a cleaner path from demo to hosted delivery."),
        ],
        MARGIN,
        322,
        CONTENT_W,
    )
    section_band(
        c,
        "Typical integrations",
        "LLM providers, document rendering, workspace files, knowledge collections, custom business tools, notifications, and MCP-backed services can all fit into the same workflow model.",
        x=MARGIN,
        y_top=110,
        w=CONTENT_W,
        h=58,
    )
    c.showPage()

    # Page 7
    new_page(c, 7)
    y = draw_title(
        c,
        "Implementation model, deliverables, and buyer handoff",
        "Prospective clients need clarity on what happens after purchase. The strongest sales document shows the delivery path, required inputs, and the assets they receive.",
        page_label="Delivery Model",
    )
    draw_flow(
        c,
        [
            ("1. Discovery", "Map workflow goals, stakeholders, data sources, tools, model constraints, and success criteria."),
            ("2. Prototype", "Configure the product, connect priority capabilities, and stand up the first end-to-end workflow."),
            ("3. Hardening", "Add retrieval, memory, validation, governance, error handling, and deployment configuration."),
            ("4. Handoff", "Provide walkthrough, environment notes, screenshots, backlog recommendations, and next-phase options."),
        ],
        x=MARGIN,
        y_top=y - 8,
        w=CONTENT_W,
    )
    info_card(
        c,
        MARGIN,
        332,
        250,
        126,
        "What the client should provide",
        "Target workflow description, business rules, sample inputs and outputs, tool access requirements, security constraints, hosting preference, and the preferred LLM provider if one already exists.",
        accent=ACCENT,
    )
    info_card(
        c,
        PAGE_W / 2 + 8,
        332,
        250,
        126,
        "What the client receives",
        "Configured codebase, tailored workflows, setup documentation, environment checklist, demo screenshots, delivery walkthrough, and a practical roadmap for additional workflow coverage.",
        accent=ACCENT_2,
    )
    section_band(
        c,
        "Recommended engagement framing",
        "Sell this as a custom implementation of a configurable product, not as a one-off prompt pack. That makes the value clearer, supports larger project scope, and aligns buyer expectations with maintainable delivery.",
        x=MARGIN,
        y_top=166,
        w=CONTENT_W,
        h=72,
    )
    c.showPage()

    # Page 8
    new_page(c, 8)
    y = draw_title(
        c,
        "How to position the offer for prospective clients",
        "The document should close by making the offer easy to understand: what it solves, who it fits, and why the client should start with this product base instead of commissioning an unstructured build.",
        page_label="Sales Positioning",
    )
    panel(c, MARGIN, y - 8, CONTENT_W, 182, fill=SURFACE)
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(MARGIN + 16, y - 30, "Suggested offer summary")
    offer_copy = (
        "I build custom agentic AI workflow systems for teams that need more than a chatbot. "
        "This product provides a working foundation with a visual workflow builder, chat-to-workflow intake, saved workflow management, knowledge retrieval, user context memory, and a production-oriented backend for planner-led or workflow-authored execution.\n\n"
        "The engagement starts by mapping your business workflow, tools, data sources, and delivery requirements. "
        "From there, I configure the platform, implement the needed capabilities, ship the first high-value workflow, and leave you with a product surface your team can actually use, review, and extend."
    )
    draw_paragraph(c, offer_copy, MARGIN + 16, y - 52, CONTENT_W - 32, size=10, leading=15, color=MUTED)
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(MARGIN, 290, "Quick fit guide")
    draw_comparison_row(
        c,
        MARGIN,
        272,
        CONTENT_W,
        46,
        "Best fit",
        "Teams that need repeatable workflows, traceable runs, reusable context, multi-tool execution, and a product surface for ongoing operations.",
        "Single-question chat widgets, isolated copywriting tasks, or short-lived demos that do not need workflow state or governance.",
        first_row=True,
    )
    draw_comparison_row(
        c,
        MARGIN,
        226,
        CONTENT_W,
        46,
        "Stakeholders",
        "Product owners, operations leaders, internal innovation teams, and technical founders.",
        "Use cases owned solely by marketing copy generation with no workflow or system integration needs.",
    )
    draw_comparison_row(
        c,
        MARGIN,
        180,
        CONTENT_W,
        46,
        "Typical outputs",
        "Workflow implementations, internal assistants, document pipelines, onboarding flows, support operations, and knowledge-grounded task automation.",
        "Generic prompt libraries without reusable product interfaces or runtime controls.",
    )
    section_band(
        c,
        "Call to action",
        "Share the workflow you want to automate, the tools or documents it depends on, and the main success metric. From there, the product can be tailored into a buyer-specific delivery plan.",
        x=MARGIN,
        y_top=110,
        w=CONTENT_W,
        h=64,
    )
    c.save()


if __name__ == "__main__":
    build()
