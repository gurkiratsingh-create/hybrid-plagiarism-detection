from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from xml.sax.saxutils import escape


ROOT = Path(r"C:\Users\GURKIRAT SINGH\OneDrive\Desktop\2nd\research_paper\plagarism_system")
OUT = ROOT / "Final_Project_Presentation.pptx"

SLIDES = [
    (
        "A Hybrid Multi-Stage Plagiarism Detection Framework Combining Lexical and Semantic Similarity",
        [
            "Submitted in partial fulfillment for the award of the degree of Bachelor of Engineering",
            "Computer Science and Engineering",
            "Submitted by: Gurkirat Singh Bhangoo, Sparsh Tyagi, Ashmit Saini, Litesh Goyal",
            "Under the Supervision of: Ms. Ankita Thakur",
            "Department of AIT-CSE, Chandigarh University",
        ],
    ),
    (
        "Outline",
        [
            "Introduction to Project",
            "Problem Formulation",
            "Objectives of the Work",
            "Methodology Used",
            "Results and Outputs",
            "Conclusion",
            "Future Scope",
            "References",
        ],
    ),
    (
        "Introduction to Project",
        [
            "Plagiarism detection is important in academic and digital environments.",
            "Traditional systems work well for exact copying but struggle with paraphrased plagiarism.",
            "Modern rewriting tools can preserve meaning while changing surface wording.",
            "The project therefore combines lexical and semantic NLP techniques.",
        ],
    ),
    (
        "Introduction to Project",
        [
            "The system combines exact matching, TF-IDF, and Sentence-BERT.",
            "Exact matching detects copied sentences.",
            "TF-IDF captures lexical similarity based on important word overlap.",
            "SBERT captures semantic similarity even when wording changes.",
            "The complete pipeline was built in Python and tested on a PAN dataset subset.",
        ],
    ),
    (
        "Project Evolution",
        [
            "Initial version: high recall but weak precision.",
            "Improved version: safer document-level aggregation.",
            "Research-ready version: PAN metadata-based pairing and threshold tuning.",
            "Reviewer-ready version: learned classifier, ablation study, and confidence intervals.",
            "Final version: synchronized code, paper, README, report, and presentation.",
        ],
    ),
    (
        "Problem Formulation",
        [
            "TF-IDF is precise for copied text but weak for paraphrasing.",
            "SBERT understands meaning but can produce false positives for topically similar documents.",
            "The early hybrid model depended too much on one strong sentence match.",
            "Main problem: improve precision without losing recall.",
            "Research requirement: make the evaluation transparent and reproducible.",
        ],
    ),
    (
        "Objectives of the Work",
        [
            "Build a full plagiarism detection pipeline using exact, lexical, and semantic similarity.",
            "Detect copied as well as paraphrased plagiarism.",
            "Improve precision while preserving high recall.",
            "Use fair evaluation with threshold tuning and reproducible train-test split.",
            "Prepare the project for publication, placement, and interview discussion.",
        ],
    ),
    (
        "Methodology Used",
        [
            "Input documents are normalized and split into sentences.",
            "Exact sentence matches are checked first.",
            "Document-level TF-IDF similarity is computed.",
            "Document-level SBERT similarity is computed.",
            "Sentence-level semantic matching is used for local evidence.",
        ],
    ),
    (
        "Methodology Used",
        [
            "Top-K matching keeps the best candidate sentence pairs.",
            "Weak sentence pairs are removed through lexical-semantic filtering.",
            "Local sentence evidence is combined with global TF-IDF and global SBERT.",
            "A tuned threshold converts the final score into plagiarized or non-plagiarized.",
            "Separate thresholds are tuned for TF-IDF, SBERT, Hybrid, and Learned Classifier.",
        ],
    ),
    (
        "Dataset and Evaluation Protocol",
        [
            "Dataset source: PAN plagiarism dataset subset.",
            "Total document pairs: 162.",
            "Training pairs: 113.",
            "Test pairs: 49.",
            "Test positives: 21, Test negatives: 28.",
            "Bootstrap samples: 500.",
        ],
    ),
    (
        "Initial vs Final Performance",
        [
            "Initial Hybrid: Precision 0.55, Recall 0.96, F1 0.69, Accuracy 0.58.",
            "Final Hybrid: Precision 0.9091, Recall 0.9524, F1 0.9302, Accuracy 0.9388.",
            "The main improvement was much higher precision while keeping recall high.",
            "This happened because the final model uses stronger aggregation, filtering, and threshold calibration.",
        ],
    ),
    (
        "Final Model Comparison",
        [
            "TF-IDF: Precision 0.6897, Recall 0.9524, F1 0.8000, Accuracy 0.7959.",
            "SBERT: Precision 0.7619, Recall 0.7619, F1 0.7619, Accuracy 0.7959.",
            "Hybrid: Precision 0.9091, Recall 0.9524, F1 0.9302, Accuracy 0.9388.",
            "Learned Classifier: Precision 0.9048, Recall 0.9048, F1 0.9048, Accuracy 0.9184.",
            "The Hybrid model gave the best overall performance.",
        ],
    ),
    (
        "Ablation Study",
        [
            "Full Hybrid F1: 0.9302.",
            "Minus Local Signal F1: 0.9302.",
            "Minus Global TF-IDF F1: 0.8182.",
            "Minus Global SBERT F1: 0.8235.",
            "Local Only F1: 0.6000.",
            "Insight: global lexical and semantic signals contributed most on this split.",
        ],
    ),
    (
        "Confidence and Error Analysis",
        [
            "Hybrid F1 bootstrap confidence interval: 0.8333 to 1.0000.",
            "The test set contains only 49 pairs, so results are promising but not definitive.",
            "Hybrid confusion matrix: TP 20, FP 2, FN 1, TN 26.",
            "The final system greatly reduced false positives compared with the early prototype.",
        ],
    ),
    (
        "Major Updates Completed",
        [
            "PAN loader updated to use metadata-based positive pairs.",
            "Unsafe max-score document aggregation was replaced.",
            "Hybrid score now combines local and global evidence.",
            "Experiment runner now uses fixed seed and threshold tuning.",
            "Learning-based classifier, ablation study, and confidence intervals were added.",
            "Paper, README, graphs, final report, and PPT were aligned.",
        ],
    ),
    (
        "Conclusion",
        [
            "A complete hybrid plagiarism detection system was successfully developed.",
            "The final model combines exact matching, TF-IDF, SBERT, top-K matching, filtering, and ensemble aggregation.",
            "The model improved from prototype-level precision to a strong final result.",
            "Final Hybrid: Precision 0.9091, Recall 0.9524, F1 0.9302, Accuracy 0.9388.",
            "The project is now stronger for publication, viva, placement, and interview explanation.",
        ],
    ),
    (
        "Future Scope",
        [
            "Evaluate on larger PAN subsets or the full benchmark protocol.",
            "Use harder negatives from realistic retrieval pipelines.",
            "Extend the system to multilingual plagiarism detection.",
            "Add explainable sentence-match outputs.",
            "Explore stronger trainable models over hybrid features.",
            "Deploy the project as a Flask API or web demo.",
        ],
    ),
    (
        "References",
        [
            "Potthast et al. - Evaluation framework for plagiarism detection.",
            "Barron-Cedeno et al. - Plagiarism and paraphrasing.",
            "Reimers and Gurevych - Sentence-BERT.",
            "Devlin et al. - BERT.",
            "Arabi and Akbari - Hybrid weighted similarity.",
            "PAN 2025 overview and participant work.",
            "Recent comparative plagiarism detection studies.",
        ],
    ),
]


def tx_box(text, x, y, cx, cy, size=2400, bold=False, color="1F1F1F", align="l"):
    runs = []
    for line in str(text).split("\n"):
        runs.append(
            f"""
            <a:p>
              <a:pPr algn="{align}"/>
              <a:r>
                <a:rPr lang="en-US" sz="{size}" {'b="1"' if bold else ''}>
                  <a:solidFill><a:srgbClr val="{color}"/></a:solidFill>
                  <a:latin typeface="Times New Roman"/>
                </a:rPr>
                <a:t>{escape(line)}</a:t>
              </a:r>
            </a:p>"""
        )
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{abs(hash((text,x,y))) % 100000 + 10}" name="TextBox"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
      <p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>
      <p:txBody><a:bodyPr wrap="square"/><a:lstStyle/>{''.join(runs)}</p:txBody>
    </p:sp>"""


def rect(x, y, cx, cy, color, line="FFFFFF"):
    return f"""
    <p:sp>
      <p:nvSpPr><p:cNvPr id="{abs(hash((x,y,cx,cy,color))) % 100000 + 100}" name="Rectangle"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
      <p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="{color}"/></a:solidFill><a:ln><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln></p:spPr>
      <p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody>
    </p:sp>"""


def slide_xml(title, bullets, number):
    is_title = number == 1
    shapes = []
    if is_title:
        shapes.append(rect(0, 0, 12192000, 850000, "1B3E7A", "1B3E7A"))
        shapes.append(tx_box(title, 700000, 1350000, 10800000, 1350000, 2600, True, "1B3E7A", "ctr"))
        shapes.append(tx_box("\n".join(bullets), 1200000, 3300000, 9800000, 2300000, 1900, False, "1F1F1F", "ctr"))
    else:
        shapes.append(rect(0, 0, 12192000, 680000, "1B3E7A", "1B3E7A"))
        shapes.append(tx_box(title, 360000, 120000, 10500000, 400000, 2700, True, "FFFFFF"))
        bullet_text = "\n".join([f"• {b}" for b in bullets])
        shapes.append(tx_box(bullet_text, 700000, 1100000, 10800000, 5200000, 2100, False, "1F1F1F"))
    shapes.append(tx_box(str(number), 11500000, 6500000, 300000, 200000, 1300, True, "1B3E7A", "ctr"))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    {''.join(shapes)}
  </p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def write_package():
    slide_ids = "\n".join(
        [
            f'<p:sldId id="{256 + i}" r:id="rId{i}"/>'
            for i in range(1, len(SLIDES) + 1)
        ]
    )
    pres_rels = "\n".join(
        [
            f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
            for i in range(1, len(SLIDES) + 1)
        ]
        + [
            f'<Relationship Id="rId{len(SLIDES)+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>'
        ]
    )
    overrides = "\n".join(
        [
            f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
            for i in range(1, len(SLIDES) + 1)
        ]
    )

    with ZipFile(OUT, "w", ZIP_DEFLATED) as z:
        z.writestr(
            "[Content_Types].xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  {overrides}
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>""",
        )
        z.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>""",
        )
        z.writestr(
            "ppt/presentation.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <p:sldSz cx="12192000" cy="6858000" type="screen16x9"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:sldIdLst>{slide_ids}</p:sldIdLst>
</p:presentation>""",
        )
        z.writestr(
            "ppt/_rels/presentation.xml.rels",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{pres_rels}
</Relationships>""",
        )
        z.writestr(
            "ppt/theme/theme1.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Office Theme">
  <a:themeElements>
    <a:clrScheme name="Office">
      <a:dk1><a:srgbClr val="000000"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="1F1F1F"/></a:dk2><a:lt2><a:srgbClr val="EEECE1"/></a:lt2>
      <a:accent1><a:srgbClr val="1B3E7A"/></a:accent1><a:accent2><a:srgbClr val="E8F0FA"/></a:accent2>
      <a:accent3><a:srgbClr val="9BBB59"/></a:accent3><a:accent4><a:srgbClr val="8064A2"/></a:accent4>
      <a:accent5><a:srgbClr val="4BACC6"/></a:accent5><a:accent6><a:srgbClr val="F79646"/></a:accent6>
      <a:hlink><a:srgbClr val="0000FF"/></a:hlink><a:folHlink><a:srgbClr val="800080"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="Office"><a:majorFont><a:latin typeface="Times New Roman"/></a:majorFont><a:minorFont><a:latin typeface="Times New Roman"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="Office"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
</a:theme>""",
        )
        for i, (title, bullets) in enumerate(SLIDES, 1):
            z.writestr(f"ppt/slides/slide{i}.xml", slide_xml(title, bullets, i))
            z.writestr(
                f"ppt/slides/_rels/slide{i}.xml.rels",
                """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"/>""",
            )
        z.writestr(
            "docProps/app.xml",
            f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"><Application>Microsoft PowerPoint</Application><Slides>{len(SLIDES)}</Slides></Properties>""",
        )
        z.writestr(
            "docProps/core.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/"><dc:title>Final Project Presentation</dc:title><dc:creator>Gurkirat Singh Bhangoo</dc:creator></cp:coreProperties>""",
        )


if __name__ == "__main__":
    write_package()
    print(OUT)
