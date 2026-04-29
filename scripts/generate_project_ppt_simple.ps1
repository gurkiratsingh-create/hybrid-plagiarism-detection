$ErrorActionPreference = "Stop"

$projectRoot = "C:\Users\GURKIRAT SINGH\OneDrive\Desktop\2nd\research_paper\plagarism_system"
$outputPath = Join-Path $projectRoot "Final_Project_Presentation.pptx"

function Rgb($r, $g, $b) {
    return ($b -shl 16) -bor ($g -shl 8) -bor $r
}

$navy = Rgb 27 62 122
$light = Rgb 232 240 250
$text = Rgb 31 31 31
$white = Rgb 255 255 255

$slidesData = @(
    @{Title="A Hybrid Multi-Stage Plagiarism Detection Framework Combining Lexical and Semantic Similarity"; Body=@("Submitted in partial fulfillment for the award of the degree of Bachelor of Engineering", "Computer Science and Engineering", "Submitted by: Gurkirat Singh Bhangoo, Sparsh Tyagi, Ashmit Saini, Litesh Goyal", "Under the Supervision of: Ms. Ankita Thakur", "Department of AIT-CSE, Chandigarh University")},
    @{Title="Outline"; Body=@("Introduction to Project", "Problem Formulation", "Objectives of the Work", "Methodology Used", "Results and Outputs", "Conclusion", "Future Scope", "References")},
    @{Title="Introduction to Project"; Body=@("Plagiarism detection is important in academic and digital environments.", "Traditional systems work well for exact copying but struggle with paraphrased plagiarism.", "Modern rewriting tools can preserve meaning while changing surface wording.", "The project combines lexical and semantic NLP techniques.")},
    @{Title="Project Idea"; Body=@("Exact matching detects directly copied sentences.", "TF-IDF captures lexical similarity through important word overlap.", "SBERT captures semantic similarity even when wording changes.", "The full system was built in Python and tested on a PAN dataset subset.")},
    @{Title="Project Evolution"; Body=@("Initial version: high recall but weak precision.", "Improved version: safer document-level aggregation.", "Research-ready version: PAN metadata-based pairing and threshold tuning.", "Reviewer-ready version: learned classifier, ablation study, and confidence intervals.", "Final version: synchronized code, paper, README, report, and presentation.")},
    @{Title="Problem Formulation"; Body=@("TF-IDF is precise for copied text but weak for paraphrasing.", "SBERT understands meaning but can produce false positives for topically similar documents.", "The early hybrid model depended too much on one strong sentence match.", "Main problem: improve precision without losing recall.", "Research requirement: make the evaluation transparent and reproducible.")},
    @{Title="Objectives of the Work"; Body=@("Build a full plagiarism detection pipeline using exact, lexical, and semantic similarity.", "Detect copied as well as paraphrased plagiarism.", "Improve precision while preserving high recall.", "Use fair evaluation with threshold tuning and reproducible train-test split.", "Prepare the project for publication, placement, and interview discussion.")},
    @{Title="Methodology Used"; Body=@("Input documents are normalized and split into sentences.", "Exact sentence matches are checked first.", "Document-level TF-IDF similarity is computed.", "Document-level SBERT similarity is computed.", "Sentence-level semantic matching is used for local evidence.")},
    @{Title="Hybrid Scoring Pipeline"; Body=@("Top-K matching keeps the best candidate sentence pairs.", "Weak sentence pairs are removed through lexical-semantic filtering.", "Local sentence evidence is combined with global TF-IDF and global SBERT.", "A tuned threshold converts the final score into plagiarized or non-plagiarized.", "Separate thresholds are tuned for TF-IDF, SBERT, Hybrid, and Learned Classifier.")},
    @{Title="Dataset and Evaluation Protocol"; Body=@("Dataset source: PAN plagiarism dataset subset.", "Total document pairs: 162.", "Training pairs: 113.", "Test pairs: 49.", "Test positives: 21 and test negatives: 28.", "Bootstrap samples: 500.")},
    @{Title="Initial vs Final Performance"; Body=@("Initial Hybrid: Precision 0.55, Recall 0.96, F1 0.69, Accuracy 0.58.", "Final Hybrid: Precision 0.9091, Recall 0.9524, F1 0.9302, Accuracy 0.9388.", "Main improvement: much higher precision while keeping recall high.", "Reason: stronger aggregation, filtering, and threshold calibration.")},
    @{Title="Final Model Comparison"; Body=@("TF-IDF: Precision 0.6897, Recall 0.9524, F1 0.8000, Accuracy 0.7959.", "SBERT: Precision 0.7619, Recall 0.7619, F1 0.7619, Accuracy 0.7959.", "Hybrid: Precision 0.9091, Recall 0.9524, F1 0.9302, Accuracy 0.9388.", "Learned Classifier: Precision 0.9048, Recall 0.9048, F1 0.9048, Accuracy 0.9184.", "Hybrid model gave the best overall performance.")},
    @{Title="Ablation Study"; Body=@("Full Hybrid F1: 0.9302.", "Minus Local Signal F1: 0.9302.", "Minus Global TF-IDF F1: 0.8182.", "Minus Global SBERT F1: 0.8235.", "Local Only F1: 0.6000.", "Insight: global lexical and semantic signals contributed most on this split.")},
    @{Title="Confidence and Error Analysis"; Body=@("Hybrid F1 bootstrap confidence interval: 0.8333 to 1.0000.", "The test set contains only 49 pairs, so results are promising but not definitive.", "Hybrid confusion matrix: TP 20, FP 2, FN 1, TN 26.", "The final system greatly reduced false positives compared with the early prototype.")},
    @{Title="Major Updates Completed"; Body=@("PAN loader updated to use metadata-based positive pairs.", "Unsafe max-score document aggregation was replaced.", "Hybrid score now combines local and global evidence.", "Experiment runner now uses fixed seed and threshold tuning.", "Learning-based classifier, ablation study, and confidence intervals were added.", "Paper, README, graphs, final report, and PPT were aligned.")},
    @{Title="Conclusion"; Body=@("A complete hybrid plagiarism detection system was successfully developed.", "The final model combines exact matching, TF-IDF, SBERT, top-K matching, filtering, and ensemble aggregation.", "The model improved from prototype-level precision to a strong final result.", "Final Hybrid: Precision 0.9091, Recall 0.9524, F1 0.9302, Accuracy 0.9388.", "The project is stronger for publication, viva, placement, and interview explanation.")},
    @{Title="Future Scope"; Body=@("Evaluate on larger PAN subsets or the full benchmark protocol.", "Use harder negatives from realistic retrieval pipelines.", "Extend the system to multilingual plagiarism detection.", "Add explainable sentence-match outputs.", "Explore stronger trainable models over hybrid features.", "Deploy the project as a Flask API or web demo.")},
    @{Title="References"; Body=@("Potthast et al. - Evaluation framework for plagiarism detection.", "Barron-Cedeno et al. - Plagiarism and paraphrasing.", "Reimers and Gurevych - Sentence-BERT.", "Devlin et al. - BERT.", "Arabi and Akbari - Hybrid weighted similarity.", "PAN 2025 overview and recent comparative plagiarism detection studies.")}
)

$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = -1
$pres = $ppt.Presentations.Add()
$pres.PageSetup.SlideSize = 16

$slideW = $pres.PageSetup.SlideWidth
$slideH = $pres.PageSetup.SlideHeight

for ($i = 0; $i -lt $slidesData.Count; $i++) {
    $slide = $pres.Slides.Add($i + 1, 12)
    $slide.FollowMasterBackground = 0
    $slide.Background.Fill.ForeColor.RGB = $white
    $slide.Background.Fill.Solid()

    if ($i -eq 0) {
        $bar = $slide.Shapes.AddShape(1, 0, 0, $slideW, 58)
        $bar.Fill.ForeColor.RGB = $navy
        $bar.Line.Visible = 0

        $title = $slide.Shapes.AddTextbox(1, 55, 105, $slideW - 110, 125)
        $title.TextFrame.TextRange.Text = $slidesData[$i].Title
        $title.TextFrame.TextRange.Font.Name = "Times New Roman"
        $title.TextFrame.TextRange.Font.Size = 25
        $title.TextFrame.TextRange.Font.Bold = -1
        $title.TextFrame.TextRange.Font.Color.RGB = $navy
        $title.TextFrame.TextRange.ParagraphFormat.Alignment = 2
        $title.Line.Visible = 0
        $title.Fill.Visible = 0

        $body = $slide.Shapes.AddShape(1, 120, 275, $slideW - 240, 260)
        $body.Fill.ForeColor.RGB = $light
        $body.Line.ForeColor.RGB = $navy
        $body.TextFrame.TextRange.Text = ($slidesData[$i].Body -join "`r")
        $body.TextFrame.TextRange.Font.Name = "Times New Roman"
        $body.TextFrame.TextRange.Font.Size = 18
        $body.TextFrame.TextRange.Font.Color.RGB = $text
        $body.TextFrame.TextRange.ParagraphFormat.Alignment = 2
    }
    else {
        $bar = $slide.Shapes.AddShape(1, 0, 0, $slideW, 52)
        $bar.Fill.ForeColor.RGB = $navy
        $bar.Line.Visible = 0

        $title = $slide.Shapes.AddTextbox(1, 28, 8, $slideW - 95, 34)
        $title.TextFrame.TextRange.Text = $slidesData[$i].Title
        $title.TextFrame.TextRange.Font.Name = "Times New Roman"
        $title.TextFrame.TextRange.Font.Size = 26
        $title.TextFrame.TextRange.Font.Bold = -1
        $title.TextFrame.TextRange.Font.Color.RGB = $white
        $title.Line.Visible = 0
        $title.Fill.Visible = 0

        $body = $slide.Shapes.AddShape(1, 60, 95, $slideW - 120, $slideH - 155)
        $body.Fill.ForeColor.RGB = $(if (($i % 2) -eq 0) { $light } else { $white })
        $body.Line.ForeColor.RGB = $navy
        $body.TextFrame.TextRange.Text = (($slidesData[$i].Body | ForEach-Object { "• $_" }) -join "`r")
        $body.TextFrame.TextRange.Font.Name = "Times New Roman"
        $body.TextFrame.TextRange.Font.Size = 21
        $body.TextFrame.TextRange.Font.Color.RGB = $text
        $body.TextFrame.TextRange.ParagraphFormat.SpaceAfter = 8
    }

    $footer = $slide.Shapes.AddTextbox(1, $slideW - 45, $slideH - 28, 35, 18)
    $footer.TextFrame.TextRange.Text = [string]($i + 1)
    $footer.TextFrame.TextRange.Font.Name = "Times New Roman"
    $footer.TextFrame.TextRange.Font.Size = 12
    $footer.TextFrame.TextRange.Font.Bold = -1
    $footer.TextFrame.TextRange.Font.Color.RGB = $navy
    $footer.Line.Visible = 0
    $footer.Fill.Visible = 0
}

if (Test-Path $outputPath) {
    Remove-Item -LiteralPath $outputPath -Force
}

$pres.SaveAs($outputPath)
$pres.Close()
$ppt.Quit()

[System.Runtime.InteropServices.Marshal]::ReleaseComObject($pres) | Out-Null
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null
[GC]::Collect()
[GC]::WaitForPendingFinalizers()

Write-Output "Saved presentation to $outputPath"
