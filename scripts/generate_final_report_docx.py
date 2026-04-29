from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape


PROJECT_ROOT = Path(r"C:\Users\GURKIRAT SINGH\OneDrive\Desktop\2nd\research_paper\plagarism_system")
TEMPLATE = Path(r"C:\Users\GURKIRAT SINGH\OneDrive\Desktop\5. Project Report_Format.docx")
OUTPUT = PROJECT_ROOT / "Final_Project_Report.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def p(text="", style=None, align=None, bold=False, italic=False, page_break=False):
    text = escape(text)
    ppr = []
    if style:
        ppr.append(f'<w:pStyle w:val="{style}"/>')
    if align:
        ppr.append(f'<w:jc w:val="{align}"/>')
    if page_break:
        ppr.append('<w:pageBreakBefore/>')
    rpr = []
    if bold:
        rpr.append("<w:b/>")
    if italic:
        rpr.append("<w:i/>")
    rpr_xml = f"<w:rPr>{''.join(rpr)}</w:rPr>" if rpr else ""
    ppr_xml = f"<w:pPr>{''.join(ppr)}</w:pPr>" if ppr else ""
    return (
        f"<w:p>{ppr_xml}<w:r>{rpr_xml}<w:t xml:space=\"preserve\">{text}</w:t></w:r></w:p>"
    )


def bullet(text):
    return p(f"• {text}", style="ListParagraph")


def chapter(number, title):
    return p(number, style="Heading1", align="center", bold=True) + p(
        title, style="Heading2", align="center", bold=True
    )


def heading(title):
    return p(title, style="Heading2", bold=True)


def subheading(title):
    return p(title, style="Heading3", bold=True)


def body(text):
    return p(text, style="BodyText")


def table(headers, rows):
    cols = len(headers)
    grid = "".join("<w:gridCol w:w=\"2200\"/>" for _ in range(cols))

    def cell(text, header=False):
        text = escape(str(text))
        bold_xml = "<w:rPr><w:b/></w:rPr>" if header else ""
        return (
            "<w:tc>"
            "<w:tcPr><w:tcW w:w=\"2200\" w:type=\"dxa\"/></w:tcPr>"
            f"<w:p><w:pPr><w:jc w:val=\"center\"/></w:pPr><w:r>{bold_xml}<w:t xml:space=\"preserve\">{text}</w:t></w:r></w:p>"
            "</w:tc>"
        )

    trs = ["<w:tr>" + "".join(cell(h, True) for h in headers) + "</w:tr>"]
    for row in rows:
        trs.append("<w:tr>" + "".join(cell(v) for v in row) + "</w:tr>")
    return (
        "<w:tbl>"
        "<w:tblPr>"
        "<w:tblStyle w:val=\"TableGrid\"/>"
        "<w:tblW w:w=\"0\" w:type=\"auto\"/>"
        "<w:tblBorders>"
        "<w:top w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:left w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:right w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"6\" w:space=\"0\" w:color=\"auto\"/>"
        "<w:insideV w:val=\"single\" w:sz=\"6\" w:space=\"0\" w:color=\"auto\"/>"
        "</w:tblBorders>"
        "</w:tblPr>"
        f"<w:tblGrid>{grid}</w:tblGrid>"
        + "".join(trs)
        + "</w:tbl>"
    )


sections = []

# Cover
sections += [
    p("A HYBRID MULTI-STAGE PLAGIARISM DETECTION FRAMEWORK COMBINING LEXICAL AND SEMANTIC SIMILARITY", style="Title", align="center", bold=True),
    p("A PROJECT REPORT", style="Heading2", align="center", bold=True),
    p("Submitted by", style="Heading4", align="center", bold=True, italic=True),
    p("Gurkirat Singh Bhangoo\nSparsh Tyagi\nAshmit Saini\nLitesh Goyal".replace("\n", " "), style="Heading2", align="center", bold=True),
    p("in partial fulfillment for the award of the degree of", style="Heading4", align="center", italic=True),
    p("BACHELOR OF ENGINEERING", style="Heading2", align="center", bold=True),
    p("IN", align="center"),
    p("COMPUTER SCIENCE AND ENGINEERING", style="Heading5", align="center", bold=True),
    p("Chandigarh University", style="Heading6", align="center"),
    p("APRIL 2026", style="Heading5", align="center", bold=True),
]

# Certificate
sections += [
    p("", page_break=True),
    p("BONAFIDE CERTIFICATE", style="Heading2", align="center", bold=True),
    body('Certified that this project report titled "A Hybrid Multi-Stage Plagiarism Detection Framework Combining Lexical and Semantic Similarity" is the bonafide work of Gurkirat Singh Bhangoo, Sparsh Tyagi, Ashmit Saini, and Litesh Goyal, who carried out the project work under the guidance and supervision of Ms. Ankita Thakur.'),
    p(""),
    p("SIGNATURE", style="Heading6", bold=True),
    p("HEAD OF THE DEPARTMENT"),
    p("SIGNATURE", style="Heading6", align="right", bold=True),
    p("Ms. Ankita Thakur", align="right"),
    p("SUPERVISOR", style="Heading6", align="right", bold=True),
]

# Acknowledgement
sections += [
    p("", page_break=True),
    p("ACKNOWLEDGEMENT", style="Heading2", align="center", bold=True),
    body("We express our sincere gratitude to our supervisor, Ms. Ankita Thakur, for her guidance, encouragement, and valuable suggestions throughout the development of this project. Her feedback helped us improve both the technical quality of the plagiarism detection system and the quality of the final research documentation."),
    body("We also thank the faculty members of the Department of Computer Science and Engineering, Chandigarh University, for providing the academic environment and resources required for this work. We are grateful to our peers for the discussions and support that improved our experimentation, evaluation, and presentation."),
    body("Finally, we acknowledge the PAN dataset contributors and the open-source NLP community for the datasets, models, and software tools that supported this project."),
]

# TOC / lists
sections += [
    p("", page_break=True),
    p("TABLE OF CONTENTS", style="Heading2", align="center", bold=True),
    body("Chapter 1. Introduction"),
    body("Chapter 2. Literature Survey"),
    body("Chapter 3. System Design, Methodology, and Version-Wise Updates"),
    body("Chapter 4. Results, Analysis, and Validation"),
    body("Chapter 5. Conclusion and Future Work"),
    body("References"),
    body("Appendix A. User Manual"),
    body("Appendix B. Achievements"),
    p("", page_break=True),
    p("LIST OF FIGURES", style="Heading3", align="center", bold=True),
    body("Figure 1. Hybrid plagiarism detection pipeline"),
    body("Figure 2. Comparative model performance across final models"),
    body("Figure 3. Hybrid model ablation study"),
    body("Figure 4. Hybrid confusion matrix on the held-out test split"),
    p("LIST OF TABLES", style="Heading3", align="center", bold=True),
    body("Table 1. Project evolution from initial version to final version"),
    body("Table 2. Train-test protocol and dataset composition"),
    body("Table 3. Final performance comparison across models"),
    body("Table 4. Ablation study of hybrid components"),
    body("Table 5. Interview- and deployment-ready project assets"),
]

# Abstract / abbreviations / symbols
sections += [
    p("", page_break=True),
    p("ABSTRACT", style="Heading2", align="center", bold=True),
    body("This report presents the complete development journey of a research-oriented plagiarism detection system that combines exact matching, lexical similarity, and semantic similarity in a single hybrid pipeline. The work began as a document-pair classifier using exact overlap, TF-IDF, and Sentence-BERT sentence matching on a controlled PAN subset. The initial version produced high recall but low precision because the document decision was overly influenced by isolated high-similarity sentence pairs."),
    body("The system was progressively redesigned to become more reliable, reproducible, and industry-ready. Major improvements included PAN metadata-based pair construction, stricter sentence-level filtering, document-level lexical and semantic aggregation, threshold calibration for each model, a reproducible train-test protocol, bootstrap confidence intervals, a learning-based comparison model, and an ablation study."),
    body("On the final held-out test split of 49 document pairs, the improved hybrid model achieved precision of 0.9091, recall of 0.9524, F1-score of 0.9302, and accuracy of 0.9388, outperforming the TF-IDF-only and SBERT-only baselines."),
    p("ABBREVIATIONS", style="Heading2", align="center", bold=True),
    bullet("AI - Artificial Intelligence"),
    bullet("BERT - Bidirectional Encoder Representations from Transformers"),
    bullet("CI - Confidence Interval"),
    bullet("NLP - Natural Language Processing"),
    bullet("PAN - Plagiarism Analysis, Authorship Identification, and Near-Duplicate Detection"),
    bullet("SBERT - Sentence-BERT"),
    bullet("TF-IDF - Term Frequency-Inverse Document Frequency"),
    p("SYMBOLS", style="Heading2", align="center", bold=True),
    bullet("w_e - weight assigned to exact-match evidence"),
    bullet("w_p - weight assigned to lexical similarity"),
    bullet("w_s - weight assigned to semantic similarity"),
    bullet("alpha, beta, gamma - document-level aggregation weights"),
    bullet("K - number of top sentence candidates retained per source sentence"),
]

# Chapter 1
sections += [
    p("", page_break=True),
    chapter("CHAPTER 1", "INTRODUCTION"),
    heading("1.1 Project Background"),
    body("Plagiarism detection is no longer limited to finding directly copied passages. Modern plagiarism can involve paraphrasing, sentence restructuring, synonym substitution, and meaning-preserving rewriting. This makes the problem suitable for hybrid natural language processing systems that combine word-overlap evidence with semantic understanding."),
    body("The project was initially designed as a research-level plagiarism detector for academic text comparison. Over time, the scope expanded to include research publication, reviewer preparedness, and industry-readiness for machine learning engineering interviews."),
    heading("1.2 Identification of Need"),
    body("The key need behind the project was to detect plagiarism reliably in settings where copied content may not preserve exact wording. A lexical-only system often misses paraphrases, while a semantic-only system may overestimate similarity between texts that discuss related topics."),
    heading("1.3 Problem Statement"),
    body("The problem addressed in this project is the accurate identification of plagiarism in document pairs that may contain exact copying, near-copying, or paraphrased content. The challenge is to improve precision without sacrificing recall while also maintaining a transparent and reproducible evaluation setup."),
    heading("1.4 Objectives"),
    bullet("Build a full plagiarism detection pipeline in Python using exact, lexical, and semantic similarity signals."),
    bullet("Improve precision without losing the strong recall observed in the early prototype."),
    bullet("Make the system research-ready through fairer experiments, ablation studies, and confidence analysis."),
    bullet("Make the project industry-ready through reproducibility and stronger explainability."),
    heading("1.5 Timeline and Development Stages"),
    table(
        ["Stage", "Main Focus", "Outcome"],
        [
            ["Version 1", "Initial hybrid prototype", "High recall but weak precision"],
            ["Version 2", "Safer aggregation and stricter filtering", "Reduced false positives"],
            ["Version 3", "Correct PAN pairing and threshold tuning", "Fairer evaluation"],
            ["Version 4", "Learning-based classifier and ablation", "Reviewer-ready evidence"],
            ["Version 5", "Paper, README, graphs, and report alignment", "Publication and interview readiness"],
        ],
    ),
]

# Chapter 2
sections += [
    p("", page_break=True),
    chapter("CHAPTER 2", "LITERATURE SURVEY"),
    heading("2.1 Evolution of Plagiarism Detection Research"),
    body("Early plagiarism detection systems mainly depended on lexical overlap, string matching, and information retrieval methods. These methods worked well for copied text but performed poorly when plagiarism was expressed through paraphrasing. Later research introduced semantic embeddings, contextual models, and transformer-based representations."),
    heading("2.2 Important Prior Work"),
    bullet("Potthast et al. established a foundational evaluation framework for plagiarism detection and PAN benchmarking."),
    bullet("Barron-Cedeno et al. highlighted the importance of paraphrasing-aware detection."),
    bullet("Reimers and Gurevych introduced Sentence-BERT for strong sentence-level semantic comparison."),
    bullet("Arabi and Akbari demonstrated the usefulness of weighted hybrid similarity for extrinsic plagiarism detection."),
    bullet("Recent PAN 2025 work confirmed that semantic alignment and retrieval-stage filtering remain central to current systems."),
    heading("2.3 Gap Identified from Literature"),
    body("The literature shows that hybrid lexical-semantic approaches are effective, but many implementations do not clearly explain how local sentence evidence and global document evidence should be combined. Another common gap is the lack of transparent reporting about threshold calibration, dataset construction, and uncertainty when experiments are run on smaller subsets."),
    heading("2.4 How This Project Responds to the Gap"),
    body("This project addresses the above gap by implementing a structured multi-stage pipeline that combines exact matching, document-level TF-IDF, document-level SBERT, top-K sentence matching, lexical-semantic filtering, and a final ensemble score."),
]

# Chapter 3
sections += [
    p("", page_break=True),
    chapter("CHAPTER 3", "SYSTEM DESIGN, METHODOLOGY, AND VERSION-WISE UPDATES"),
    heading("3.1 Initial System Architecture"),
    body("The first version of the system used preprocessing, sentence splitting, exact matching, TF-IDF similarity, and SBERT similarity. Candidate sentence pairs were scored and the highest sentence-level score often influenced the final document decision. This gave the system very high recall, but it also created many false positives."),
    heading("3.2 Core Components of the Final Pipeline"),
    bullet("Text preprocessing and sentence segmentation"),
    bullet("Exact match detection for copied content"),
    bullet("Document-level TF-IDF similarity for lexical comparison"),
    bullet("Document-level SBERT similarity for semantic comparison"),
    bullet("Top-K sentence candidate generation for efficient local matching"),
    bullet("Lexical-semantic filtering to remove weak or noisy sentence pairs"),
    bullet("Document-level aggregation combining local and global evidence"),
    bullet("Threshold calibration and classification"),
    heading("3.3 Version-by-Version Technical Updates"),
    subheading("3.3.1 Version 1: Prototype Hybrid Model"),
    body("The original prototype combined exact matching, TF-IDF, and SBERT. It successfully detected many positives, but aggregation relied too heavily on the strongest sentence match. Evaluation was also limited because a shared threshold was used across all methods."),
    subheading("3.3.2 Version 2: Safer Aggregation and Filtering"),
    body("The next update redesigned the hybrid aggregation logic. Instead of using a single sentence maximum, the system began combining top non-overlapping sentence matches with document-level similarity signals. Weak sentence pairs were filtered more carefully, which improved precision and reduced noisy positives."),
    subheading("3.3.3 Version 3: Data and Evaluation Corrections"),
    body("The PAN loader was updated to use PAN metadata for positive-pair construction instead of positional file matching. Negative pairs were sampled from suspicious-source combinations that were not linked in the PAN pairs file. The experiment runner was rewritten to use a fixed random seed, a reproducible train-test split, and separate threshold tuning for TF-IDF, SBERT, and the hybrid model."),
    subheading("3.3.4 Version 4: Reviewer-Facing Enhancements"),
    body("The project was extended with a learning-based logistic-regression classifier built on extracted hybrid features. An ablation study was added to measure the role of each signal, and bootstrap confidence intervals were introduced to acknowledge uncertainty caused by the modest test size."),
    subheading("3.3.5 Version 5: Publication and Presentation Alignment"),
    body("The final update synchronized the codebase, experiment outputs, graphs, README, and LaTeX paper. The paper was revised to cite more recent literature, reduce overclaiming, and explicitly discuss limitations such as sampled negatives and the controlled subset."),
    heading("3.4 Important Code-Level Improvements"),
    table(
        ["Component", "Before", "After"],
        [
            ["Dataset loader", "Position assumptions", "PAN metadata-based positives and sampled negatives"],
            ["Hybrid aggregation", "One strong sentence could dominate", "Local and global evidence combined"],
            ["Experiment thresholds", "One threshold for all methods", "Independent threshold tuning"],
            ["Evaluation", "Prototype-level metrics", "Reproducible split, CI, and ablation"],
            ["Model scope", "Heuristic hybrid only", "Hybrid plus learned classifier"],
        ],
    ),
]

# Chapter 4
sections += [
    p("", page_break=True),
    chapter("CHAPTER 4", "RESULTS, ANALYSIS, AND VALIDATION"),
    heading("4.1 Dataset and Evaluation Protocol"),
    table(
        ["Item", "Value"],
        [
            ["Dataset source", "PAN plagiarism dataset subset"],
            ["Total pairs", "162"],
            ["Training pairs", "113"],
            ["Test pairs", "49"],
            ["Test positives", "21"],
            ["Test negatives", "28"],
            ["Random seed", "42"],
            ["Bootstrap samples", "500"],
        ],
    ),
    heading("4.2 Initial Performance vs Final Performance"),
    body("The original paper-stage hybrid system reported precision near 0.55, recall near 0.96, F1 near 0.69, and accuracy near 0.58. These results showed that the system was sensitive but not selective enough. After the redesign and evaluation improvements, the final hybrid system achieved much stronger precision while preserving high recall."),
    table(
        ["System Version", "Precision", "Recall", "F1", "Accuracy"],
        [
            ["Initial hybrid version", "0.55", "0.96", "0.69", "0.58"],
            ["Final hybrid version", "0.9091", "0.9524", "0.9302", "0.9388"],
        ],
    ),
    heading("4.3 Final Model Comparison"),
    table(
        ["Method", "Precision", "Recall", "F1", "Accuracy"],
        [
            ["TF-IDF", "0.6897", "0.9524", "0.8000", "0.7959"],
            ["SBERT", "0.7619", "0.7619", "0.7619", "0.7959"],
            ["Hybrid", "0.9091", "0.9524", "0.9302", "0.9388"],
            ["Learned Classifier", "0.9048", "0.9048", "0.9048", "0.9184"],
        ],
    ),
    heading("4.4 Ablation Study"),
    body("The ablation study was added to answer a key reviewer question: which parts of the hybrid system actually matter? The results showed that the global lexical and semantic signals contributed most of the discriminative power on the current split, while the local sentence-level signal added comparatively little."),
    table(
        ["Variant", "F1 Score"],
        [
            ["Full Hybrid", "0.9302"],
            ["Minus Local Signal", "0.9302"],
            ["Minus Global TF-IDF", "0.8182"],
            ["Minus Global SBERT", "0.8235"],
            ["Local Only", "0.6000"],
        ],
    ),
    heading("4.5 Confidence and Error Analysis"),
    body("Because the held-out test set contains only 49 document pairs, statistical confidence is important. The hybrid model achieved an F1-score confidence interval of approximately 0.8333 to 1.0000 under bootstrap resampling. This supports the conclusion that the hybrid model is strong on the selected subset, while still showing that larger-scale testing is needed."),
    body("The final hybrid model produced only two false positives and one false negative on the held-out split. This is a large improvement over the early prototype."),
]

# Chapter 5
sections += [
    p("", page_break=True),
    chapter("CHAPTER 5", "CONCLUSION AND FUTURE WORK"),
    heading("5.1 Conclusion"),
    body("This project started as a promising but prototype-level plagiarism detector and matured into a much stronger hybrid framework. The final system combines exact matching, document-level TF-IDF, document-level SBERT, top-K sentence matching, and lexical-semantic filtering in a single calibrated pipeline."),
    body("The most important achievement is that precision improved dramatically while recall remained high. The final hybrid model outperformed TF-IDF-only and SBERT-only baselines and stayed competitive even when compared with a learning-based classifier built from extracted hybrid features."),
    heading("5.2 Future Work"),
    bullet("Evaluate the system on a much larger PAN subset or the full benchmark protocol."),
    bullet("Use harder negatives obtained through realistic retrieval rather than simple sampling."),
    bullet("Extend the system to multilingual and cross-domain plagiarism detection."),
    bullet("Explore stronger trainable models over the hybrid features."),
    bullet("Add explanation outputs that highlight the sentence pairs responsible for a decision."),
    heading("5.3 Practical Outcome"),
    table(
        ["Asset", "Current Status"],
        [
            ["Hybrid pipeline code", "Updated and working"],
            ["Reproducible experiment runner", "Updated and working"],
            ["Graphs and visualizations", "Updated from current comparison.json"],
            ["README", "Professional and synchronized"],
            ["Research paper", "Revised with stronger positioning and references"],
            ["Final report", "Generated in university-style format"],
        ],
    ),
]

# References and appendices
refs = [
    "Arabi, H. and Akbari, M. (2022). Improving plagiarism detection in text document using hybrid weighted similarity. Expert Systems with Applications, 207, 118034.",
    "Barron-Cedeno, A., Vila, M., Marti, M. A., and Rosso, P. (2013). Plagiarism meets paraphrasing: Insights for the next generation in automatic plagiarism detection. Computational Linguistics, 39(4), 917-947.",
    "Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. NAACL-HLT 2019.",
    "Foltynek, T., Meuschke, N., and Gipp, B. (2019). Academic plagiarism detection: A systematic literature review. ACM Computing Surveys, 52(6), 112.",
    "Greiner-Petter, A., Frobe, M., Wahle, J. P., Ruas, T., Gipp, B., Aizawa, A., and Potthast, M. (2025). Overview of the plagiarism detection task at PAN 2025. CLEF 2025 Working Notes, CEUR-WS Vol. 4038.",
    "Luo, J., Huang, M., Liu, B., and Han, Z. (2025). Team JR at generative plagiarism detection 2025: A two-stage approach from TF-IDF/Jaccard filtering to transformer classification. CLEF 2025 Working Notes, CEUR-WS Vol. 4038.",
    "Mehdi, M., Mushtaq, S., and Butt, G. R. (2025). A hybrid TF-IDF and SBERT approach for enhanced text classification performance. Preprints.org.",
    "Potthast, M., Stein, B., Barron-Cedeno, A., and Rosso, P. (2010). An evaluation framework for plagiarism detection. COLING 2010.",
    "Pudasaini, S., Miralles-Pechuan, L., Lillis, D., and Llorens Salvador, M. (2024). Survey on plagiarism detection in large language models: The impact of ChatGPT and Gemini on academic integrity. arXiv:2407.13105.",
    "Reimers, N. and Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using siamese BERT-networks. EMNLP-IJCNLP 2019.",
    "Sajid, M., Sanaullah, M., Fuzail, M., Malik, T. S., and Shuhidan, S. M. (2025). Comparative analysis of text-based plagiarism detection techniques. PLOS ONE, 20(4), e0319551.",
    "Salton, G. and Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. Information Processing and Management, 24(5), 513-523.",
]
sections += [p("", page_break=True), p("REFERENCES", style="Heading2", align="center", bold=True)]
sections += [body(r) for r in refs]
sections += [
    p("", page_break=True),
    p("APPENDIX A - USER MANUAL", style="Heading2", align="center", bold=True),
    bullet("Step 1: Open the project folder in the workspace."),
    bullet("Step 2: Run the main experiment using python -m experiments.run_experiment."),
    bullet("Step 3: Check results/comparison.json for saved metrics and thresholds."),
    bullet("Step 4: Regenerate graphs using the plotting script if needed."),
    bullet("Step 5: Use Docs/Latex/newrp.tex to keep the paper aligned with the latest results."),
    p("APPENDIX B - ACHIEVEMENTS", style="Heading2", align="center", bold=True),
    bullet("Developed a full hybrid plagiarism detection pipeline using exact match, TF-IDF, and SBERT."),
    bullet("Improved the project from prototype-level performance to a much stronger final hybrid result."),
    bullet("Added reproducible evaluation, threshold calibration, bootstrap confidence intervals, and ablation study."),
    bullet("Prepared publication-aligned assets including a revised paper, updated figures, and a professional README."),
]


sect_pr = (
    "<w:sectPr>"
    "<w:pgSz w:w=\"11906\" w:h=\"16838\"/>"
    "<w:pgMar w:top=\"1440\" w:right=\"1134\" w:bottom=\"1440\" w:left=\"1701\" w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/>"
    "</w:sectPr>"
)

document_xml = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    f'<w:document xmlns:w="{W_NS}"><w:body>'
    + "".join(sections)
    + sect_pr
    + "</w:body></w:document>"
)


def build_docx():
    with ZipFile(TEMPLATE, "r") as src, ZipFile(OUTPUT, "w", ZIP_DEFLATED) as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            if info.filename == "word/document.xml":
                data = document_xml.encode("utf-8")
            dst.writestr(info, data)


if __name__ == "__main__":
    build_docx()
    print(f"Saved report to {OUTPUT}")
