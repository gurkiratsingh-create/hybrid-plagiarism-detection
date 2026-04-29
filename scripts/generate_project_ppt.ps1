$ErrorActionPreference = "Stop"

$projectRoot = "C:\Users\GURKIRAT SINGH\OneDrive\Desktop\2nd\research_paper\plagarism_system"
$outputPath = Join-Path $projectRoot "Final_Project_Presentation.pptx"
$assets = Join-Path $projectRoot "Assets"

$pipelineImage = Join-Path $assets "pipeline.png"
$modelImage = Join-Path $assets "model_comparison.png"
$ablationImage = Join-Path $assets "ablation_study.png"
$confusionImage = Join-Path $assets "hybrid_confusion_matrix.png"

$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = -1
$pres = $ppt.Presentations.Add()
$pres.PageSetup.SlideSize = 16

$slideW = $pres.PageSetup.SlideWidth
$slideH = $pres.PageSetup.SlideHeight

$titleColor = 0x7A3E1B   # RGB(27,62,122)
$accentColor = 0xD6E3F3  # RGB(243,227,214) reversed by Office BGR? We'll set via helper.
$textColor = 0x1F1F1F
$mutedColor = 0x666666
$footerColor = 0x7A3E1B
$bgColor = 0xFFFFFF

function Set-RgbColor {
    param([int]$R, [int]$G, [int]$B)
    return ($B -shl 16) -bor ($G -shl 8) -bor $R
}

$titleColor = Set-RgbColor 27 62 122
$accentColor = Set-RgbColor 232 240 250
$textColor = Set-RgbColor 31 31 31
$mutedColor = Set-RgbColor 102 102 102
$lineColor = Set-RgbColor 210 220 235

function Set-SlideBackground {
    param($Slide)
    $Slide.FollowMasterBackground = 0
    $Slide.Background.Fill.ForeColor.RGB = $bgColor
    $Slide.Background.Fill.Solid()
}

function Add-Header {
    param($Slide, [string]$Title, [int]$PageNo)
    Set-SlideBackground $Slide
    $bar = $Slide.Shapes.AddShape(1, 0, 0, $slideW, 55)
    $bar.Fill.ForeColor.RGB = $titleColor
    $bar.Line.Visible = 0

    $titleBox = $Slide.Shapes.AddTextbox(1, 28, 8, $slideW - 120, 36)
    $titleBox.TextFrame.TextRange.Text = $Title
    $titleBox.TextFrame.TextRange.Font.Name = "Times New Roman"
    $titleBox.TextFrame.TextRange.Font.Size = 28
    $titleBox.TextFrame.TextRange.Font.Bold = -1
    $titleBox.TextFrame.TextRange.Font.Color.RGB = $bgColor
    $titleBox.Line.Visible = 0
    $titleBox.Fill.Visible = 0

    $footer = $Slide.Shapes.AddTextbox(1, $slideW - 50, $slideH - 28, 30, 20)
    $footer.TextFrame.TextRange.Text = [string]$PageNo
    $footer.TextFrame.TextRange.Font.Name = "Times New Roman"
    $footer.TextFrame.TextRange.Font.Size = 14
    $footer.TextFrame.TextRange.Font.Bold = -1
    $footer.TextFrame.TextRange.Font.Color.RGB = $footerColor
    $footer.Line.Visible = 0
    $footer.Fill.Visible = 0
}

function Add-BulletsBox {
    param($Slide, [string[]]$Bullets, [double]$Left = 60, [double]$Top = 95, [double]$Width = 1160, [double]$Height = 540, [int]$FontSize = 22)
    $box = $Slide.Shapes.AddTextbox(1, $Left, $Top, $Width, $Height)
    $box.Line.Visible = 0
    $box.Fill.Visible = 0
    $textRange = $box.TextFrame.TextRange
    $textRange.Text = ($Bullets -join "`r")
    $textRange.Font.Name = "Times New Roman"
    $textRange.Font.Size = $FontSize
    $textRange.Font.Color.RGB = $textColor
    $textRange.ParagraphFormat.Bullet.Visible = -1
    $textRange.ParagraphFormat.SpaceAfter = 6
    return $box
}

function Add-ParagraphBox {
    param($Slide, [string[]]$Paragraphs, [double]$Left = 60, [double]$Top = 95, [double]$Width = 1160, [double]$Height = 540, [int]$FontSize = 22)
    $box = $Slide.Shapes.AddTextbox(1, $Left, $Top, $Width, $Height)
    $box.Line.Visible = 0
    $box.Fill.Visible = 0
    $textRange = $box.TextFrame.TextRange
    $textRange.Text = ($Paragraphs -join "`r`r")
    $textRange.Font.Name = "Times New Roman"
    $textRange.Font.Size = $FontSize
    $textRange.Font.Color.RGB = $textColor
    return $box
}

function Add-Image {
    param($Slide, [string]$Path, [double]$Left, [double]$Top, [double]$Width, [double]$Height)
    if (Test-Path $Path) {
        $pic = $Slide.Shapes.AddPicture($Path, 0, -1, $Left, $Top, $Width, $Height)
        return $pic
    }
}

function Add-TitleSlide {
    param($Slide, [int]$PageNo)
    Set-SlideBackground $Slide
    $banner = $Slide.Shapes.AddShape(1, 0, 0, $slideW, 82)
    $banner.Fill.ForeColor.RGB = $titleColor
    $banner.Line.Visible = 0

    $title = $Slide.Shapes.AddTextbox(1, 80, 120, 1120, 120)
    $title.TextFrame.TextRange.Text = "A Hybrid Multi-Stage Plagiarism Detection Framework Combining Lexical and Semantic Similarity"
    $title.TextFrame.TextRange.Font.Name = "Times New Roman"
    $title.TextFrame.TextRange.Font.Size = 26
    $title.TextFrame.TextRange.Font.Bold = -1
    $title.TextFrame.TextRange.ParagraphFormat.Alignment = 2
    $title.Line.Visible = 0
    $title.Fill.Visible = 0

    $subtitle = $Slide.Shapes.AddTextbox(1, 150, 270, 980, 70)
    $subtitle.TextFrame.TextRange.Text = "Submitted in the partial fulfillment for the award of the degree of`rBachelor of Engineering in Computer Science and Engineering"
    $subtitle.TextFrame.TextRange.Font.Name = "Times New Roman"
    $subtitle.TextFrame.TextRange.Font.Size = 20
    $subtitle.TextFrame.TextRange.ParagraphFormat.Alignment = 2
    $subtitle.Line.Visible = 0
    $subtitle.Fill.Visible = 0

    $leftCard = $Slide.Shapes.AddShape(1, 110, 395, 420, 180)
    $leftCard.Fill.ForeColor.RGB = $accentColor
    $leftCard.Line.ForeColor.RGB = $lineColor
    $leftCard.TextFrame.TextRange.Text = "Submitted by:`rGurkirat Singh Bhangoo`rSparsh Tyagi`rAshmit Saini`rLitesh Goyal"
    $leftCard.TextFrame.TextRange.Font.Name = "Times New Roman"
    $leftCard.TextFrame.TextRange.Font.Size = 20
    $leftCard.TextFrame.TextRange.Font.Bold = -1

    $rightCard = $Slide.Shapes.AddShape(1, 700, 395, 420, 180)
    $rightCard.Fill.ForeColor.RGB = $accentColor
    $rightCard.Line.ForeColor.RGB = $lineColor
    $rightCard.TextFrame.TextRange.Text = "Under the Supervision of:`rMs. Ankita Thakur`r`rDepartment of AIT-CSE`rChandigarh University"
    $rightCard.TextFrame.TextRange.Font.Name = "Times New Roman"
    $rightCard.TextFrame.TextRange.Font.Size = 20
    $rightCard.TextFrame.TextRange.Font.Bold = -1

    $footer = $Slide.Shapes.AddTextbox(1, $slideW - 50, $slideH - 28, 30, 20)
    $footer.TextFrame.TextRange.Text = [string]$PageNo
    $footer.TextFrame.TextRange.Font.Name = "Times New Roman"
    $footer.TextFrame.TextRange.Font.Size = 14
    $footer.TextFrame.TextRange.Font.Bold = -1
    $footer.TextFrame.TextRange.Font.Color.RGB = $footerColor
    $footer.Line.Visible = 0
    $footer.Fill.Visible = 0
}

function Add-TableSlide {
    param($Slide, [string]$Title, [object[][]]$Data, [string[]]$Headers, [int]$PageNo)
    Add-Header $Slide $Title $PageNo
    $rows = $Data.Count + 1
    $cols = $Headers.Count
    $table = $Slide.Shapes.AddTable($rows, $cols, 90, 120, 1100, 430).Table
    for ($c=1; $c -le $cols; $c++) {
        $table.Cell(1,$c).Shape.TextFrame.TextRange.Text = $Headers[$c-1]
        $table.Cell(1,$c).Shape.Fill.ForeColor.RGB = $titleColor
        $table.Cell(1,$c).Shape.TextFrame.TextRange.Font.Name = "Times New Roman"
        $table.Cell(1,$c).Shape.TextFrame.TextRange.Font.Size = 16
        $table.Cell(1,$c).Shape.TextFrame.TextRange.Font.Bold = -1
        $table.Cell(1,$c).Shape.TextFrame.TextRange.Font.Color.RGB = $bgColor
    }
    for ($r=0; $r -lt $Data.Count; $r++) {
        for ($c=0; $c -lt $cols; $c++) {
            $cell = $table.Cell($r+2, $c+1)
            $cell.Shape.TextFrame.TextRange.Text = [string]$Data[$r][$c]
            $cell.Shape.TextFrame.TextRange.Font.Name = "Times New Roman"
            $cell.Shape.TextFrame.TextRange.Font.Size = 15
            $cell.Shape.Fill.ForeColor.RGB = $(if (($r % 2) -eq 0) { $accentColor } else { $bgColor })
        }
    }
}

function Add-ImageSlide {
    param($Slide, [string]$Title, [string[]]$Bullets, [string]$ImagePath, [int]$PageNo)
    Add-Header $Slide $Title $PageNo
    Add-BulletsBox $Slide $Bullets 60 110 470 500 20 | Out-Null
    $frame = $Slide.Shapes.AddShape(1, 575, 120, 620, 430)
    $frame.Fill.ForeColor.RGB = $accentColor
    $frame.Line.ForeColor.RGB = $lineColor
    Add-Image $Slide $ImagePath 590 135 590 400 | Out-Null
}

# Slide list
$slides = @()
for ($i = 1; $i -le 18; $i++) { $slides += $pres.Slides.Add($i, 12) }

Add-TitleSlide $slides[0] 1

Add-Header $slides[1] "Outline" 2
Add-BulletsBox $slides[1] @(
    "Introduction to Project",
    "Problem Formulation",
    "Objectives of the Work",
    "Methodology Used",
    "Results and Outputs",
    "Conclusion",
    "Future Scope",
    "References"
) 120 120 900 380 26 | Out-Null

Add-Header $slides[2] "Introduction to Project" 3
Add-BulletsBox $slides[2] @(
    "Plagiarism detection is important in academic and digital environments.",
    "Traditional systems work well for exact copying but struggle with paraphrased plagiarism.",
    "Modern rewriting tools can preserve meaning while changing surface wording.",
    "This makes plagiarism detection a strong use case for hybrid NLP systems."
) 80 120 1100 420 24 | Out-Null

Add-Header $slides[3] "Introduction to Project" 4
Add-BulletsBox $slides[3] @(
    "The project combines three levels of evidence:",
    "Exact matching for copied sentences",
    "TF-IDF for lexical similarity",
    "SBERT for semantic similarity",
    "The complete pipeline was built in Python and evaluated on a PAN dataset subset."
) 80 120 1100 420 24 | Out-Null

Add-Header $slides[4] "Introduction to Project" 5
Add-BulletsBox $slides[4] @(
    "The work evolved from a research prototype into a stronger ML-engineering style system.",
    "Key upgrades included metadata-based PAN pairing, better aggregation, threshold tuning, reproducible experiments, a learned classifier, ablation study, and confidence intervals.",
    "The final goal was to make the project stronger for publication, interviews, and technical discussion."
) 80 120 1100 430 22 | Out-Null

Add-Header $slides[5] "Problem Formulation" 6
Add-BulletsBox $slides[5] @(
    "TF-IDF is precise for copied text but weak for paraphrasing.",
    "SBERT captures meaning better but can increase false positives for topically similar text.",
    "The early hybrid prototype had high recall but low precision.",
    "Main challenge: improve precision without losing recall while keeping the evaluation scientifically correct."
) 80 120 1100 420 24 | Out-Null

Add-Header $slides[6] "Objectives of the Work" 7
Add-BulletsBox $slides[6] @(
    "Build a full plagiarism detection pipeline using exact, lexical, and semantic similarity signals.",
    "Detect both copied and paraphrased plagiarism.",
    "Improve precision without collapsing recall.",
    "Make the system research-ready through fair evaluation and analysis.",
    "Make the project industry-ready through reproducibility and explainability."
) 80 120 1100 430 23 | Out-Null

Add-ImageSlide $slides[7] "Methodology Used" @(
    "Input documents are preprocessed and split into sentences.",
    "Exact sentence matches are checked first.",
    "Whole-document TF-IDF and SBERT similarities are computed.",
    "Sentence-level matching is then used for local evidence."
) $pipelineImage 8

Add-Header $slides[8] "Methodology Used" 9
Add-BulletsBox $slides[8] @(
    "Top-K matching keeps only the best candidate sentence pairs.",
    "Weak pairs are removed using lexical-semantic filtering.",
    "Local evidence is combined with global TF-IDF and global SBERT.",
    "The final hybrid score is passed through a tuned threshold for classification."
) 80 120 1100 430 24 | Out-Null

Add-TableSlide $slides[9] "Methodology Used" @(
    @("Dataset source","PAN plagiarism subset"),
    @("Total pairs","162"),
    @("Train size","113"),
    @("Test size","49"),
    @("Test positives","21"),
    @("Test negatives","28"),
    @("Bootstrap samples","500")
) @("Item","Value") 10

Add-TableSlide $slides[10] "Results and Outputs" @(
    @("Initial hybrid version","0.55","0.96","0.69","0.58"),
    @("Final hybrid version","0.9091","0.9524","0.9302","0.9388")
) @("System Version","Precision","Recall","F1","Accuracy") 11

Add-TableSlide $slides[11] "Results and Outputs" @(
    @("TF-IDF","0.6897","0.9524","0.8000","0.7959"),
    @("SBERT","0.7619","0.7619","0.7619","0.7959"),
    @("Hybrid","0.9091","0.9524","0.9302","0.9388"),
    @("Learned Classifier","0.9048","0.9048","0.9048","0.9184")
) @("Method","Precision","Recall","F1","Accuracy") 12

Add-ImageSlide $slides[12] "Results and Outputs" @(
    "The hybrid model gave the best overall balance across precision, recall, F1, and accuracy.",
    "TF-IDF remained strong for copied content.",
    "SBERT alone was weaker when used as the only signal.",
    "The learned classifier was competitive but did not beat the final hybrid."
) $modelImage 13

Add-ImageSlide $slides[13] "Results and Outputs" @(
    "Ablation Study F1 Scores:",
    "Full Hybrid: 0.9302",
    "Minus Local Signal: 0.9302",
    "Minus Global TF-IDF: 0.8182",
    "Minus Global SBERT: 0.8235",
    "Local Only: 0.6000"
) $ablationImage 14

Add-ImageSlide $slides[14] "Results and Outputs" @(
    "Major project upgrades:",
    "PAN metadata-based pairing",
    "Better aggregation beyond a single best sentence",
    "Independent threshold tuning",
    "Bootstrap confidence intervals",
    "Learning-based classifier and ablation analysis"
) $confusionImage 15

Add-Header $slides[15] "Conclusion" 16
Add-BulletsBox $slides[15] @(
    "A hybrid plagiarism detection system was successfully developed.",
    "The final system combines exact matching, TF-IDF, SBERT, top-K sentence matching, filtering, and ensemble aggregation.",
    "The project improved from a prototype with low precision to a much stronger final model.",
    "Final hybrid performance: Precision 0.9091, Recall 0.9524, F1 0.9302, Accuracy 0.9388."
) 80 120 1100 420 23 | Out-Null

Add-Header $slides[16] "Future Scope" 17
Add-BulletsBox $slides[16] @(
    "Evaluate on larger PAN subsets or the full benchmark setting.",
    "Use harder negatives from realistic retrieval pipelines.",
    "Extend the system to multilingual plagiarism detection.",
    "Add explanation outputs showing suspicious sentence matches.",
    "Explore stronger trainable models over the hybrid feature set.",
    "Deploy the project as a Flask/API or web demo."
) 80 120 1100 430 23 | Out-Null

Add-Header $slides[17] "References" 18
Add-BulletsBox $slides[17] @(
    "Potthast et al. - evaluation framework for plagiarism detection",
    "Barron-Cedeno et al. - plagiarism and paraphrasing",
    "Reimers and Gurevych - Sentence-BERT",
    "Devlin et al. - BERT",
    "Arabi and Akbari - hybrid weighted similarity",
    "PAN 2025 overview and participant work",
    "Recent comparative plagiarism detection studies"
) 70 110 1120 460 20 | Out-Null

$pres.SaveAs($outputPath)
$pres.Close()
$ppt.Quit()

[System.Runtime.InteropServices.Marshal]::ReleaseComObject($pres) | Out-Null
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null

Write-Output "Saved presentation to $outputPath"
