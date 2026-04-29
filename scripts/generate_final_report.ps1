$ErrorActionPreference = "Stop"

$projectRoot = "C:\Users\GURKIRAT SINGH\OneDrive\Desktop\2nd\research_paper\plagarism_system"
$outputPath = Join-Path $projectRoot "Final_Project_Report.docx"

$pipelineImage = Join-Path $projectRoot "pipeline.png"
$modelImage = Join-Path $projectRoot "model_comparison.png"
$ablationImage = Join-Path $projectRoot "ablation_study.png"
$confusionImage = Join-Path $projectRoot "hybrid_confusion_matrix.png"

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$doc = $word.Documents.Add()

function Add-Paragraph {
    param(
        [string]$Text,
        [string]$Style = "",
        [int]$Size = 12,
        [bool]$Bold = $false,
        [bool]$Italic = $false,
        [int]$Alignment = 0,
        [double]$SpaceAfter = 6,
        [double]$LineSpacing = 18
    )

    $p = $doc.Content.Paragraphs.Add()
    if ($Style) {
        try { $p.Range.Style = $Style } catch {}
    }
    $p.Range.Text = $Text
    $p.Range.Font.Name = "Times New Roman"
    $p.Range.Font.Size = $Size
    $p.Range.Font.Bold = [int]$Bold
    $p.Range.Font.Italic = [int]$Italic
    $p.Alignment = $Alignment
    $p.SpaceAfter = $SpaceAfter
    $p.LineSpacingRule = 4
    $p.LineSpacing = $LineSpacing
    $p.Range.InsertParagraphAfter() | Out-Null
    return $p
}

function Add-PageBreak {
    $range = $doc.Content
    $range.Collapse(0)
    $range.InsertBreak(7) | Out-Null
}

function Add-ChapterHeading {
    param([string]$NumberText, [string]$Title)
    Add-Paragraph -Text $NumberText -Style "Heading 1" -Size 16 -Bold $true -Alignment 1 -SpaceAfter 3 | Out-Null
    Add-Paragraph -Text $Title -Style "Heading 1" -Size 16 -Bold $true -Alignment 1 -SpaceAfter 12 | Out-Null
}

function Add-SectionHeading {
    param([string]$Title)
    Add-Paragraph -Text $Title -Style "Heading 2" -Size 14 -Bold $true -Alignment 0 -SpaceAfter 6 | Out-Null
}

function Add-SubSectionHeading {
    param([string]$Title)
    Add-Paragraph -Text $Title -Style "Heading 3" -Size 12 -Bold $true -Alignment 0 -SpaceAfter 4 | Out-Null
}

function Add-Body {
    param([string]$Text)
    Add-Paragraph -Text $Text -Style "Normal" -Size 12 -Alignment 3 -SpaceAfter 6 | Out-Null
}

function Add-Bullets {
    param([string[]]$Items)
    foreach ($item in $Items) {
        $p = Add-Paragraph -Text $item -Style "Normal" -Size 12 -Alignment 3 -SpaceAfter 3
        try { $p.Range.ListFormat.ApplyBulletDefault() } catch {}
    }
}

function Add-TableFromData {
    param(
        [string[]]$Headers,
        [object[][]]$Rows
    )
    $range = $doc.Content
    $range.Collapse(0)
    $table = $doc.Tables.Add($range, $Rows.Count + 1, $Headers.Count)
    $table.Range.Font.Name = "Times New Roman"
    $table.Range.Font.Size = 11
    $table.Borders.Enable = 1
    for ($c = 1; $c -le $Headers.Count; $c++) {
        $table.Cell(1, $c).Range.Text = $Headers[$c - 1]
        $table.Cell(1, $c).Range.Bold = 1
        $table.Cell(1, $c).Range.ParagraphFormat.Alignment = 1
    }
    for ($r = 0; $r -lt $Rows.Count; $r++) {
        for ($c = 0; $c -lt $Headers.Count; $c++) {
            $table.Cell($r + 2, $c + 1).Range.Text = [string]$Rows[$r][$c]
            $table.Cell($r + 2, $c + 1).Range.ParagraphFormat.Alignment = 1
        }
    }
    $doc.Content.InsertParagraphAfter() | Out-Null
    return $table
}

function Add-ImageWithCaption {
    param(
        [string]$Path,
        [string]$Caption,
        [double]$Width = 400
    )
    if (-not (Test-Path $Path)) { return }
    $range = $doc.Content
    $range.Collapse(0)
    $shape = $doc.InlineShapes.AddPicture($Path, $false, $true, $range)
    $shape.Width = $Width
    $shape.Range.ParagraphFormat.Alignment = 1
    $doc.Content.InsertParagraphAfter() | Out-Null
    Add-Paragraph -Text $Caption -Size 10 -Bold $true -Alignment 1 -SpaceAfter 10 -LineSpacing 12 | Out-Null
}

function Set-PageNumbering {
    foreach ($section in $doc.Sections) {
        $footer = $section.Footers.Item(1)
        $footer.PageNumbers.Add(2) | Out-Null
        $footer.Range.Font.Name = "Times New Roman"
        $footer.Range.Font.Size = 10
    }
}

# A4 page setup
$doc.PageSetup.PaperSize = 7
$doc.PageSetup.TopMargin = $word.CentimetersToPoints(2.54)
$doc.PageSetup.BottomMargin = $word.CentimetersToPoints(2.54)
$doc.PageSetup.LeftMargin = $word.CentimetersToPoints(3.0)
$doc.PageSetup.RightMargin = $word.CentimetersToPoints(2.5)

# Cover Page
Add-Paragraph -Text "A HYBRID MULTI-STAGE PLAGIARISM DETECTION FRAMEWORK COMBINING LEXICAL AND SEMANTIC SIMILARITY" -Size 18 -Bold $true -Alignment 1 -SpaceAfter 24 | Out-Null
Add-Paragraph -Text "A PROJECT REPORT" -Size 14 -Bold $true -Alignment 1 -SpaceAfter 18 | Out-Null
Add-Paragraph -Text "Submitted by" -Size 14 -Bold $true -Italic $true -Alignment 1 -SpaceAfter 12 | Out-Null
Add-Paragraph -Text "Gurkirat Singh Bhangoo`r`nSparsh Tyagi`r`nAshmit Saini`r`nLitesh Goyal" -Size 16 -Bold $true -Alignment 1 -SpaceAfter 18 | Out-Null
Add-Paragraph -Text "in partial fulfillment for the award of the degree of" -Size 14 -Italic $true -Alignment 1 -SpaceAfter 12 | Out-Null
Add-Paragraph -Text "BACHELOR OF ENGINEERING" -Size 16 -Bold $true -Alignment 1 -SpaceAfter 8 | Out-Null
Add-Paragraph -Text "IN" -Size 14 -Alignment 1 -SpaceAfter 8 | Out-Null
Add-Paragraph -Text "COMPUTER SCIENCE AND ENGINEERING" -Size 14 -Bold $true -Alignment 1 -SpaceAfter 18 | Out-Null
Add-Paragraph -Text "Chandigarh University" -Size 14 -Alignment 1 -SpaceAfter 8 | Out-Null
Add-Paragraph -Text "APRIL 2026" -Size 14 -Bold $true -Alignment 1 -SpaceAfter 12 | Out-Null

Add-PageBreak

# Bonafide Certificate
Add-Paragraph -Text "BONAFIDE CERTIFICATE" -Size 16 -Bold $true -Alignment 1 -SpaceAfter 16 | Out-Null
Add-Body "Certified that this project report titled ""A Hybrid Multi-Stage Plagiarism Detection Framework Combining Lexical and Semantic Similarity"" is the bonafide work of Gurkirat Singh Bhangoo, Sparsh Tyagi, Ashmit Saini, and Litesh Goyal, who carried out the project work under the guidance and supervision of Ms. Ankita Thakur."
Add-Paragraph -Text " " -SpaceAfter 30 | Out-Null
Add-Paragraph -Text "SIGNATURE" -Size 12 -Bold $true -Alignment 0 -SpaceAfter 3 | Out-Null
Add-Paragraph -Text "Head of the Department" -Size 12 -Alignment 0 -SpaceAfter 18 | Out-Null
Add-Paragraph -Text "SIGNATURE" -Size 12 -Bold $true -Alignment 2 -SpaceAfter 3 | Out-Null
Add-Paragraph -Text "Ms. Ankita Thakur`r`nSUPERVISOR" -Size 12 -Alignment 2 -SpaceAfter 18 | Out-Null
Add-Paragraph -Text "Submitted for the project viva-voce examination held on: _____________________" -Size 12 -Alignment 0 -SpaceAfter 10 | Out-Null
Add-Paragraph -Text "INTERNAL EXAMINER                                  EXTERNAL EXAMINER" -Size 12 -Bold $true -Alignment 1 -SpaceAfter 6 | Out-Null

Add-PageBreak

# Acknowledgement
Add-Paragraph -Text "ACKNOWLEDGEMENT" -Size 16 -Bold $true -Alignment 1 -SpaceAfter 14 | Out-Null
Add-Body "We express our sincere gratitude to our supervisor, Ms. Ankita Thakur, for her steady guidance, constructive suggestions, and encouragement throughout the development of this project. Her feedback helped us improve both the technical depth of the plagiarism detection system and the quality of the final research documentation."
Add-Body "We also thank the faculty members of the Department of Computer Science and Engineering, Chandigarh University, for providing the academic environment and resources required for this work. We are grateful to our classmates and peers whose discussions and feedback supported the experimentation, evaluation, and reporting stages of the project."
Add-Body "Finally, we acknowledge the PAN dataset contributors and the open-source NLP community for providing the benchmark datasets, embedding models, and software libraries that made this work possible."

Add-PageBreak

# TOC placeholder
Add-Paragraph -Text "TABLE OF CONTENTS" -Size 16 -Bold $true -Alignment 1 -SpaceAfter 10 | Out-Null
$tocRange = $doc.Content.Paragraphs.Last.Range
Add-PageBreak

# List of Figures
Add-Paragraph -Text "LIST OF FIGURES" -Size 16 -Bold $true -Alignment 1 -SpaceAfter 10 | Out-Null
Add-Bullets @(
    "Figure 1. Hybrid plagiarism detection pipeline",
    "Figure 2. Comparative model performance across precision, recall, F1-score, and accuracy",
    "Figure 3. Hybrid model ablation study",
    "Figure 4. Hybrid confusion matrix on the held-out test split"
)

Add-PageBreak

# List of Tables
Add-Paragraph -Text "LIST OF TABLES" -Size 16 -Bold $true -Alignment 1 -SpaceAfter 10 | Out-Null
Add-Bullets @(
    "Table 1. Project evolution from initial version to final version",
    "Table 2. Train-test protocol and dataset composition",
    "Table 3. Final performance comparison across models",
    "Table 4. Ablation study of hybrid components",
    "Table 5. Interview- and deployment-ready project assets"
)

Add-PageBreak

# Abstract
Add-Paragraph -Text "ABSTRACT" -Size 16 -Bold $true -Alignment 1 -SpaceAfter 12 | Out-Null
Add-Body "This project report presents the complete development journey of a research-oriented plagiarism detection system that combines exact matching, lexical similarity, and semantic similarity in a single hybrid pipeline. The work began as a document-pair classifier using exact overlap, TF-IDF, and Sentence-BERT sentence matching on a controlled PAN subset. The initial version produced high recall but low precision because the document decision was overly influenced by isolated high-similarity sentence pairs."
Add-Body "The system was progressively redesigned to become more reliable, reproducible, and industry-ready. Major improvements included PAN metadata-based pair construction, stricter sentence-level filtering, document-level lexical and semantic aggregation, threshold calibration for each model, a reproducible train-test protocol, bootstrap confidence intervals, a learning-based comparison model, and an ablation study. These updates transformed the project from a prototype into a stronger research and engineering artifact."
Add-Body "On the final held-out test split of 49 document pairs, the improved hybrid model achieved precision of 0.9091, recall of 0.9524, F1-score of 0.9302, and accuracy of 0.9388, outperforming the TF-IDF-only and SBERT-only baselines. The report explains what was built, why each update was needed, how every major version evolved, and how the final system can be presented for publication, placements, and technical interviews."

Add-PageBreak

# Graphical Abstract
Add-Paragraph -Text "GRAPHICAL ABSTRACT" -Size 16 -Bold $true -Alignment 1 -SpaceAfter 10 | Out-Null
Add-Body "The graphical abstract summarizes the final workflow of the project: preprocessing, sentence segmentation, exact matching, document-level TF-IDF scoring, document-level SBERT scoring, top-K sentence candidate generation, lexical-semantic filtering, score aggregation, and final classification."
Add-ImageWithCaption -Path $pipelineImage -Caption "Figure 1. Hybrid plagiarism detection pipeline" -Width 380

Add-PageBreak

# Abbreviations
Add-Paragraph -Text "ABBREVIATIONS" -Size 16 -Bold $true -Alignment 1 -SpaceAfter 10 | Out-Null
Add-Bullets @(
    "AI - Artificial Intelligence",
    "BERT - Bidirectional Encoder Representations from Transformers",
    "CI - Confidence Interval",
    "CMT - Conference Management Toolkit",
    "F1 - Harmonic Mean of Precision and Recall",
    "NLP - Natural Language Processing",
    "PAN - Plagiarism Analysis, Authorship Identification, and Near-Duplicate Detection",
    "SBERT - Sentence-BERT",
    "TF-IDF - Term Frequency-Inverse Document Frequency"
)

Add-PageBreak

# Symbols
Add-Paragraph -Text "SYMBOLS" -Size 16 -Bold $true -Alignment 1 -SpaceAfter 10 | Out-Null
Add-Bullets @(
    "w_e - weight assigned to exact-match evidence",
    "w_p - weight assigned to lexical similarity",
    "w_s - weight assigned to semantic similarity",
    "alpha - weight assigned to local sentence-level evidence",
    "beta - weight assigned to global TF-IDF score",
    "gamma - weight assigned to global SBERT score",
    "K - number of top semantic sentence candidates retained per source sentence"
)

Add-PageBreak

# Chapter 1
Add-ChapterHeading "CHAPTER 1" "INTRODUCTION"
Add-SectionHeading "1.1 Project Background"
Add-Body "Plagiarism detection is no longer limited to finding directly copied passages. Modern plagiarism can involve paraphrasing, sentence restructuring, synonym substitution, and meaning-preserving rewriting. This makes the problem suitable for hybrid natural language processing systems that combine word-overlap evidence with semantic understanding."
Add-Body "The project was initially designed as a research-level plagiarism detector for academic text comparison. Over time, the scope expanded to include research publication, reviewer preparedness, and industry-readiness for machine learning engineering interviews. The final system therefore needed to be accurate, explainable, reproducible, and strong enough to be presented as both a research contribution and an engineering project."

Add-SectionHeading "1.2 Identification of Need"
Add-Body "The key need behind the project was to detect plagiarism reliably in settings where copied content may not preserve exact wording. A lexical-only system often misses paraphrases, while a semantic-only system may overestimate similarity between texts that discuss related topics. A hybrid system was therefore required to improve precision and recall together."

Add-SectionHeading "1.3 Problem Statement"
Add-Body "The problem addressed in this project is the accurate identification of plagiarism in document pairs that may contain exact copying, near-copying, or paraphrased content. The challenge is to improve precision without sacrificing recall while also maintaining a transparent and reproducible evaluation setup."

Add-SectionHeading "1.4 Objectives"
Add-Bullets @(
    "Build a full plagiarism detection pipeline in Python using exact, lexical, and semantic similarity signals.",
    "Improve precision without losing the strong recall observed in the early prototype.",
    "Make the system research-ready through fairer experiments, ablation studies, and confidence analysis.",
    "Make the project industry-ready through cleaner evaluation code, reproducibility, and stronger explainability."
)

Add-SectionHeading "1.5 Timeline and Development Stages"
$timelineHeaders = @("Stage", "Main Focus", "Outcome")
$timelineRows = @(
    @("Version 1", "Initial hybrid prototype with exact match, TF-IDF, SBERT, and sentence-level scoring", "High recall but weak precision"),
    @("Version 2", "Pipeline redesign with safer aggregation and stricter filtering", "Reduced false positives"),
    @("Version 3", "Correct PAN pairing, threshold tuning, reproducible train-test split", "Fairer and stronger evaluation"),
    @("Version 4", "Learning-based classifier, ablation study, bootstrap confidence intervals", "Reviewer-ready evidence"),
    @("Version 5", "Paper, README, graphs, and final report alignment", "Publication and interview readiness")
)
Add-TableFromData -Headers $timelineHeaders -Rows $timelineRows | Out-Null

Add-SectionHeading "1.6 Organization of the Report"
Add-Body "Chapter 1 introduces the problem, need, goals, and development plan. Chapter 2 reviews relevant literature and positions the project with respect to prior work. Chapter 3 explains the architecture and tracks the system updates from the earliest version to the final one. Chapter 4 presents experiments, validation, and final outcomes. Chapter 5 concludes the project and identifies future work."

Add-PageBreak

# Chapter 2
Add-ChapterHeading "CHAPTER 2" "LITERATURE SURVEY"
Add-SectionHeading "2.1 Evolution of Plagiarism Detection Research"
Add-Body "Early plagiarism detection systems mainly depended on lexical overlap, string matching, and information retrieval methods. These methods worked well for copied text but performed poorly when plagiarism was expressed through paraphrasing. Later research introduced semantic embeddings, contextual models, and transformer-based representations that improved coverage for meaning-preserving rewrites."
Add-Body "Recent research has increasingly favored hybrid systems because lexical and semantic approaches complement each other. Lexical features are precise for copied passages, while semantic embeddings detect paraphrased similarities that lexical methods may miss."

Add-SectionHeading "2.2 Important Prior Work"
Add-Bullets @(
    "Potthast et al. established a foundational evaluation framework for plagiarism detection and PAN benchmarking.",
    "Barron-Cedeno et al. highlighted the importance of paraphrasing-aware detection.",
    "Reimers and Gurevych introduced Sentence-BERT, enabling strong sentence-level semantic comparison.",
    "Arabi and Akbari demonstrated the usefulness of weighted hybrid similarity for extrinsic plagiarism detection.",
    "Recent PAN 2025 work confirmed that semantic alignment and retrieval-stage filtering remain central to current systems."
)

Add-SectionHeading "2.3 Gap Identified from Literature"
Add-Body "The literature shows that hybrid lexical-semantic approaches are effective, but many implementations either remain task-specific, rely heavily on one signal, or do not clearly explain how local sentence evidence and global document evidence should be combined. Another common gap is the lack of transparent reporting about threshold calibration, dataset construction, and uncertainty when experiments are run on smaller subsets."

Add-SectionHeading "2.4 How This Project Responds to the Gap"
Add-Body "This project addresses the above gap by implementing a structured multi-stage pipeline that combines exact matching, document-level TF-IDF, document-level SBERT, top-K sentence matching, lexical-semantic filtering, and a final ensemble score. The project also includes threshold tuning, metadata-based pair construction, confidence intervals, and ablation analysis so that the reported improvements are easier to justify."

Add-SectionHeading "2.5 Design Constraints and Practical Considerations"
Add-Bullets @(
    "The project uses a controlled PAN subset rather than the full shared-task scale, which limits the strength of general claims.",
    "Sentence-level matching is computationally expensive, so top-K selection was introduced for efficiency.",
    "Sampled negatives can be easier than real retrieval-stage negatives, so evaluation must be interpreted carefully.",
    "The system must be understandable enough to explain to reviewers, faculty, and interviewers."
)

Add-PageBreak

# Chapter 3
Add-ChapterHeading "CHAPTER 3" "SYSTEM DESIGN, METHODOLOGY, AND VERSION-WISE UPDATES"
Add-SectionHeading "3.1 Initial System Architecture"
Add-Body "The first version of the system used preprocessing, sentence splitting, exact matching, TF-IDF similarity, and SBERT similarity. Candidate sentence pairs were scored and the highest sentence-level score often influenced the final document decision. This gave the system very high recall, but it also created many false positives because a single lucky sentence match could dominate the result."

Add-SectionHeading "3.2 Core Components of the Final Pipeline"
Add-Bullets @(
    "Text preprocessing and sentence segmentation",
    "Exact match detection for copied content",
    "Document-level TF-IDF similarity for lexical comparison",
    "Document-level SBERT similarity for semantic comparison",
    "Top-K sentence candidate generation for efficient local matching",
    "Lexical-semantic filtering to remove weak or noisy sentence pairs",
    "Document-level aggregation combining local and global evidence",
    "Threshold calibration and classification"
)

Add-ImageWithCaption -Path $pipelineImage -Caption "Figure 2. Final hybrid workflow used in the report and paper" -Width 380

Add-SectionHeading "3.3 Version-by-Version Technical Updates"

Add-SubSectionHeading "3.3.1 Version 1: Prototype Hybrid Model"
Add-Body "The original prototype combined exact matching, TF-IDF, and SBERT. It successfully detected many positives, but aggregation relied too heavily on the strongest sentence match. Evaluation was also limited because a shared threshold was used across all methods."

Add-SubSectionHeading "3.3.2 Version 2: Safer Aggregation and Filtering"
Add-Body "The next update redesigned the hybrid aggregation logic. Instead of using a single sentence maximum, the system began combining top non-overlapping sentence matches with document-level similarity signals. Weak sentence pairs were filtered more carefully, which improved precision and reduced noisy positives."

Add-SubSectionHeading "3.3.3 Version 3: Data and Evaluation Corrections"
Add-Body "The PAN loader was updated to use PAN metadata for positive-pair construction instead of positional file matching. Negative pairs were sampled from suspicious-source combinations that were not linked in the PAN pairs file. The experiment runner was rewritten to use a fixed random seed, a reproducible train-test split, and separate threshold tuning for TF-IDF, SBERT, and the hybrid model."

Add-SubSectionHeading "3.3.4 Version 4: Reviewer-Facing Enhancements"
Add-Body "To strengthen the research contribution, the project was extended with a learning-based logistic-regression classifier built on extracted hybrid features. An ablation study was added to measure the role of each signal, and bootstrap confidence intervals were introduced to acknowledge uncertainty caused by the modest test size."

Add-SubSectionHeading "3.3.5 Version 5: Publication and Presentation Alignment"
Add-Body "The final update synchronized the codebase, experiment outputs, graphs, README, and LaTeX paper. The paper was revised to cite more recent literature, reduce overclaiming, and explicitly discuss limitations such as sampled negatives, the controlled subset, and the fact that the system is not directly comparable to full PAN 2025 shared-task pipelines."

Add-SectionHeading "3.4 Important Code-Level Improvements"
$updateHeaders = @("Component", "Before", "After")
$updateRows = @(
    @("Dataset loader", "Pairs largely relied on position assumptions", "PAN metadata-based positive pairs and sampled negatives"),
    @("Hybrid aggregation", "Dominated by one strong sentence pair", "Combined local evidence with global lexical and semantic signals"),
    @("Experiment thresholds", "One threshold for all methods", "Independent threshold tuning for each method"),
    @("Evaluation", "Prototype-level metric reporting", "Reproducible split, bootstrap confidence intervals, ablation study"),
    @("Model scope", "Heuristic hybrid only", "Hybrid plus learned logistic-regression comparison")
)
Add-TableFromData -Headers $updateHeaders -Rows $updateRows | Out-Null

Add-SectionHeading "3.5 Why Each Update Was Needed"
Add-Body "Each major update addressed a specific weakness. Better pair construction improved label quality. Better aggregation reduced false positives. Threshold tuning made the comparison fairer. Ablation and confidence intervals addressed common reviewer concerns. The learned classifier showed that the engineered hybrid features are informative even when passed to a trainable model."

Add-PageBreak

# Chapter 4
Add-ChapterHeading "CHAPTER 4" "RESULTS, ANALYSIS, AND VALIDATION"
Add-SectionHeading "4.1 Dataset and Evaluation Protocol"
$protocolHeaders = @("Item", "Value")
$protocolRows = @(
    @("Dataset source", "PAN plagiarism dataset subset"),
    @("Total pairs", "162"),
    @("Training pairs", "113"),
    @("Test pairs", "49"),
    @("Test positives", "21"),
    @("Test negatives", "28"),
    @("Random seed", "42"),
    @("Bootstrap samples", "500")
)
Add-TableFromData -Headers $protocolHeaders -Rows $protocolRows | Out-Null

Add-SectionHeading "4.2 Initial Performance vs Final Performance"
Add-Body "The original paper-stage hybrid system reported precision near 0.55, recall near 0.96, F1 near 0.69, and accuracy near 0.58. These results showed that the system was sensitive but not selective enough. After the redesign and evaluation improvements, the final hybrid system achieved much stronger precision while preserving high recall."
$perfHeaders = @("System Version", "Precision", "Recall", "F1", "Accuracy")
$perfRows = @(
    @("Initial hybrid version", "0.55", "0.96", "0.69", "0.58"),
    @("Final hybrid version", "0.9091", "0.9524", "0.9302", "0.9388")
)
Add-TableFromData -Headers $perfHeaders -Rows $perfRows | Out-Null

Add-SectionHeading "4.3 Final Model Comparison"
$finalHeaders = @("Method", "Precision", "Recall", "F1", "Accuracy")
$finalRows = @(
    @("TF-IDF", "0.6897", "0.9524", "0.8000", "0.7959"),
    @("SBERT", "0.7619", "0.7619", "0.7619", "0.7959"),
    @("Hybrid", "0.9091", "0.9524", "0.9302", "0.9388"),
    @("Learned Classifier", "0.9048", "0.9048", "0.9048", "0.9184")
)
Add-TableFromData -Headers $finalHeaders -Rows $finalRows | Out-Null
Add-ImageWithCaption -Path $modelImage -Caption "Figure 3. Comparative model performance across evaluation metrics" -Width 420

Add-SectionHeading "4.4 Ablation Study"
Add-Body "The ablation study was added to answer a key reviewer question: which parts of the hybrid system actually matter? The results showed that the global lexical and semantic signals contributed most of the discriminative power on the current split, while the local sentence-level signal added comparatively little."
$ablationHeaders = @("Variant", "F1 Score")
$ablationRows = @(
    @("Full Hybrid", "0.9302"),
    @("Minus Local Signal", "0.9302"),
    @("Minus Global TF-IDF", "0.8182"),
    @("Minus Global SBERT", "0.8235"),
    @("Local Only", "0.6000")
)
Add-TableFromData -Headers $ablationHeaders -Rows $ablationRows | Out-Null
Add-ImageWithCaption -Path $ablationImage -Caption "Figure 4. Hybrid model ablation study" -Width 360

Add-SectionHeading "4.5 Confidence and Error Analysis"
Add-Body "Because the held-out test set contains only 49 document pairs, statistical confidence is important. The hybrid model achieved an F1-score confidence interval of approximately 0.8333 to 1.0000 under bootstrap resampling. This range supports the conclusion that the hybrid model is strong on the selected subset, but it also reminds us that larger-scale testing is still needed."
Add-Body "The confusion-matrix view helps explain the result practically. The final hybrid model produced only two false positives and one false negative on the held-out split. This is a large improvement over the early prototype, which often flagged non-plagiarized pairs because of isolated high semantic matches."
Add-ImageWithCaption -Path $confusionImage -Caption "Figure 5. Hybrid confusion matrix on the held-out test split" -Width 340

Add-SectionHeading "4.6 Validation of Project Goals"
Add-Bullets @(
    "Goal 1: Hybrid plagiarism pipeline built successfully - achieved.",
    "Goal 2: Precision improved without collapsing recall - achieved.",
    "Goal 3: Research readiness with ablation and confidence analysis - achieved.",
    "Goal 4: Industry-readiness through reproducible experiments and cleaner project assets - achieved."
)

Add-PageBreak

# Chapter 5
Add-ChapterHeading "CHAPTER 5" "CONCLUSION AND FUTURE WORK"
Add-SectionHeading "5.1 Conclusion"
Add-Body "This project started as a promising but prototype-level plagiarism detector and matured into a much stronger hybrid framework. The final system combines exact matching, document-level TF-IDF, document-level SBERT, top-K sentence matching, and lexical-semantic filtering in a single calibrated pipeline. The system not only improved technically, but also became more defensible as a research artifact because the evaluation setup, reporting, and limitations are now clearly stated."
Add-Body "The most important achievement is that precision improved dramatically while recall remained high. The final hybrid model outperformed TF-IDF-only and SBERT-only baselines and stayed competitive even when compared with a learning-based classifier built from extracted hybrid features."

Add-SectionHeading "5.2 Future Work"
Add-Bullets @(
    "Evaluate the system on a much larger PAN subset or the full benchmark protocol.",
    "Use harder negatives obtained through realistic retrieval rather than simple sampling.",
    "Extend the system to multilingual and cross-domain plagiarism detection.",
    "Explore stronger trainable models such as gradient boosting or neural fusion over the hybrid features.",
    "Add explanation outputs that highlight the exact sentence pairs responsible for a plagiarism decision.",
    "Package the project as a deployable API or demo interface for interview and portfolio use."
)

Add-SectionHeading "5.3 Practical Outcome"
$assetHeaders = @("Asset", "Current Status")
$assetRows = @(
    @("Hybrid pipeline code", "Updated and working"),
    @("Reproducible experiment runner", "Updated and working"),
    @("Graphs and visualizations", "Updated from current comparison.json"),
    @("README", "Professional and synchronized"),
    @("Research paper", "Revised with stronger positioning and references"),
    @("Final report", "Generated in university-style format")
)
Add-TableFromData -Headers $assetHeaders -Rows $assetRows | Out-Null

Add-PageBreak

# References
Add-Paragraph -Text "REFERENCES" -Size 16 -Bold $true -Alignment 1 -SpaceAfter 12 | Out-Null
$references = @(
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
    "Salton, G. and Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. Information Processing and Management, 24(5), 513-523."
)
foreach ($ref in $references) { Add-Body $ref }

Add-PageBreak

# Appendix A
Add-Paragraph -Text "APPENDIX A - USER MANUAL" -Size 16 -Bold $true -Alignment 1 -SpaceAfter 12 | Out-Null
Add-Bullets @(
    "Step 1: Open the project folder at C:\\Users\\GURKIRAT SINGH\\OneDrive\\Desktop\\2nd\\research_paper\\plagarism_system.",
    "Step 2: Run the main experiment using the command: python -m experiments.run_experiment",
    "Step 3: Wait for dataset loading, threshold tuning, and final model comparison to finish.",
    "Step 4: Check results/comparison.json for the saved metrics, thresholds, and ablation outputs.",
    "Step 5: Regenerate graphs using the plotting script if needed.",
    "Step 6: Use the LaTeX files in Docs\\Latex to update the research paper based on the latest results.",
    "Step 7: Review README.md for project overview, architecture, and execution notes."
)

Add-PageBreak

# Appendix B
Add-Paragraph -Text "APPENDIX B - ACHIEVEMENTS" -Size 16 -Bold $true -Alignment 1 -SpaceAfter 12 | Out-Null
Add-Bullets @(
    "Developed a full hybrid plagiarism detection pipeline using exact match, TF-IDF, and SBERT.",
    "Improved the project from prototype-level performance to a much stronger final hybrid result.",
    "Added reproducible evaluation, threshold calibration, bootstrap confidence intervals, and ablation study.",
    "Prepared publication-aligned assets including a revised paper, updated figures, and a professional README.",
    "Created a portfolio-quality project suitable for interviews, placements, and technical discussion."
)

Set-PageNumbering

# Insert automatic TOC
try {
    $doc.TablesOfContents.Add($tocRange, $true, 1, 3) | Out-Null
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
} catch {}

$doc.SaveAs([ref]$outputPath)
$doc.Close()
$word.Quit()

[System.Runtime.Interopservices.Marshal]::ReleaseComObject($doc) | Out-Null
[System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null

Write-Output "Saved report to $outputPath"
