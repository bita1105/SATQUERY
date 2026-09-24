import React, { useState } from "react";

import {
  Satellite,
  Upload,
  ArrowRight,
  Sparkles,
  Image as ImageIcon,
  BrainCircuit,
  Map,
  Activity,
  CheckCircle2,
  ChevronRight,
  FileImage,
  X,
  Radio,
  ScanLine,
  Cpu,
  Waves,
  TreePine,
  Building2,
  GitCompare,
} from "lucide-react";


// ============================================================
// TYPES
// ============================================================

type SemanticEvidence = Record<string, string>;

type SemanticChange = {
  before: string;
  after: string;
  change: string;
};

type ChangeAnalysis = {
  statistics?: {
    total_pixels: number;
    changed_pixels: number;
    changed_percentage: number;
    unchanged_percentage: number;
  };

  change_map?: string;

  image_size?: {
    width: number;
    height: number;
  };

  before_evidence?: SemanticEvidence;
  after_evidence?: SemanticEvidence;
  changes?: Record<string, SemanticChange>;
};


type GroundingBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

type GroundingDetection = {
  label: string;
  confidence: number;
  box: GroundingBox;
  area_ratio?: number;
};

type GroundingResult = {
  query: string;
  labels_searched: string[];
  detections: GroundingDetection[];
  detection_count: number;
  average_confidence?: number;
  grounding_map: string;
  image_size: {
    width: number;
    height: number;
  };
  parameters?: {
    detection_threshold?: number;
    nms_threshold?: number;
    max_detections?: number;
    max_box_area_ratio?: number;
    min_box_area_ratio?: number;
  };
};


type OpticalSarFinding = {
  feature: string;
  fused_score: number;
  level: string;
  explanation?: string;
};

type OpticalSarResult = {
  analysis_type: string;
  image_size?: {
    width: number;
    height: number;
  };
  optical?: Record<string, number>;
  sar?: Record<string, number>;
  registration?: {
    status?: string;
    optical_keypoints?: number;
    sar_keypoints?: number;
    good_matches?: number;
    inliers?: number;
    inlier_ratio?: number;
  };
  cross_modal_findings?: OpticalSarFinding[];
  confidence?: number;
  confidence_type?: string;
  confidence_label?: string;
  reliability?: string;
  warnings?: string[];
  answer?: string;
  limitations?: string[];
};


type ExecutionTraceStep = {
  step: string;
  status: string;
  task?: string;
  model?: string;
};


type AgentResult = {
  task: string;
  reason: string;
  requires_second_image: boolean;
};


type AnalysisResult = {
  status: string;

  query: string;

  detected_task: string;

  confidence: number | null;

  message: string;

  answer?: string;

  evidence?: Record<string, string>;

  image?: {
    filename: string;
    content_type: string;
    format: string;
    width: number;
    height: number;
    mode?: string;
    size_bytes?: number;
    size_mb?: number;
  };

  second_image?: {
    filename: string;
    content_type: string;
    format: string;
    width: number;
    height: number;
    size_bytes?: number;
    size_mb?: number;
  };

  validation?: {
    is_valid: boolean;
    format_supported: boolean;
    second_image_valid?: boolean;
  };

  agent?: AgentResult;

  change_analysis?: ChangeAnalysis;

  // Change VQA fields returned by the backend
  before_evidence?: SemanticEvidence;
  after_evidence?: SemanticEvidence;
  changes?: Record<string, SemanticChange>;

  // Change detection fields returned by the backend
  change_map?: string;
  changed_area_percent?: number;
  unchanged_area_percent?: number;
  changed_pixels?: number;
  total_pixels?: number;

  // Grounding fields returned by the backend
  grounding?: GroundingResult;
  grounding_map?: string;
  detections?: GroundingDetection[];
  detection_count?: number;
  labels_searched?: string[];
  average_confidence?: number;

  // Optical + SAR fields returned by the backend
  optical_sar?: OpticalSarResult;
  registration?: OpticalSarResult["registration"];
  cross_modal_findings?: OpticalSarFinding[];
  optical?: Record<string, number>;
  sar?: Record<string, number>;
  image_size?: {
    width: number;
    height: number;
  };

  execution_trace?: ExecutionTraceStep[];
};


// ============================================================
// STAR FIELD
// ============================================================

const stars = Array.from(
  { length: 70 },
  (_, i) => ({
    id: i,
    left: `${(i * 37) % 100}%`,
    top: `${(i * 61) % 100}%`,
    delay: `${(i % 8) * 0.7}s`,
    duration: `${3 + (i % 5)}s`,
  })
);


// ============================================================
// APP
// ============================================================

function App() {

  // ----------------------------------------------------------
  // IMAGE STATE
  // ----------------------------------------------------------

  const [image, setImage] =
    useState<string | null>(null);

  const [secondImage, setSecondImage] =
    useState<string | null>(null);

  const [imageFile, setImageFile] =
    useState<File | null>(null);

  const [secondImageFile, setSecondImageFile] =
    useState<File | null>(null);


  // ----------------------------------------------------------
  // QUERY / ANALYSIS STATE
  // ----------------------------------------------------------

  const [query, setQuery] =
    useState("");

  const [analyzing, setAnalyzing] =
    useState(false);

  const [analyzed, setAnalyzed] =
    useState(false);

  const [analysisResult, setAnalysisResult] =
    useState<AnalysisResult | null>(null);

  const [error, setError] =
    useState<string | null>(null);


  // ----------------------------------------------------------
  // IMAGE UPLOAD
  // ----------------------------------------------------------

  const handleImageUpload = (
    event: React.ChangeEvent<HTMLInputElement>,
    second = false
  ) => {

    const file =
      event.target.files?.[0];

    if (!file) return;

    const url =
      URL.createObjectURL(file);


    if (second) {

      if (secondImage) {
        URL.revokeObjectURL(secondImage);
      }

      setSecondImage(url);
      setSecondImageFile(file);

    } else {

      if (image) {
        URL.revokeObjectURL(image);
      }

      setImage(url);
      setImageFile(file);
    }


    setAnalyzed(false);
    setAnalysisResult(null);
    setError(null);
  };


  // ----------------------------------------------------------
  // ANALYZE
  // ----------------------------------------------------------

  const handleAnalyze = async () => {

    if (!imageFile || !query.trim()) {
      return;
    }


    setAnalyzing(true);
    setAnalyzed(false);
    setAnalysisResult(null);
    setError(null);


    try {

      const formData =
        new FormData();


      // Primary image
      formData.append(
        "image",
        imageFile
      );


      // Query
      formData.append(
        "query",
        query.trim()
      );


      // Secondary image
      // Only append if the user selected one.
      if (secondImageFile) {

        formData.append(
          "second_image",
          secondImageFile
        );
      }


      const API_BASE_URL =
        import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

      const response =
        await fetch(
          `${API_BASE_URL}/api/analyze`,
          {
            method: "POST",
            body: formData,
          }
        );


      const data =
        await response.json();


      // Normalize backend execution strings into the UI trace format.
      // The backend returns a simple string list for traceability.
      if (Array.isArray(data.execution)) {
        data.execution_trace = data.execution.map(
          (step: unknown, index: number) => {
            if (typeof step === "string") {
              return {
                step,
                status: "complete",
              };
            }

            if (step && typeof step === "object") {
              return {
                ...(step as Record<string, unknown>),
                status:
                  typeof (step as Record<string, unknown>).status === "string"
                    ? (step as Record<string, unknown>).status
                    : "complete",
              };
            }

            return {
              step: `Execution step ${index + 1}`,
              status: "complete",
            };
          }
        );
      }


      if (!response.ok) {

        throw new Error(
          data?.message ||
          `Backend request failed with status ${response.status}`
        );
      }


      // Backend can return status:error
      if (data.status === "error") {

        throw new Error(
          data.message ||
          "Analysis failed."
        );
      }


      // --------------------------------------------------------
      // NORMALIZE CHANGE ANALYSIS RESPONSE
      // --------------------------------------------------------
      // The current backend returns Change VQA and Change Detection
      // fields at the top level. The UI uses one unified change_analysis
      // object so both pipelines can render consistently.

      if (
        data.detected_task === "change_vqa" ||
        data.detected_task === "change_detection"
      ) {

        const changeVqa =
          data.change_vqa || {};

        const beforeEvidence =
          data.before_evidence ||
          changeVqa.before_evidence ||
          {};

        const afterEvidence =
          data.after_evidence ||
          changeVqa.after_evidence ||
          {};

        const semanticChanges =
          data.changes ||
          changeVqa.changes ||
          {};

        const hasPixelChangeData =
          Boolean(data.change_map) ||
          data.changed_area_percent !== undefined ||
          data.changed_pixels !== undefined;

        const hasSemanticChangeData =
          Object.keys(beforeEvidence).length > 0 ||
          Object.keys(afterEvidence).length > 0 ||
          Object.keys(semanticChanges).length > 0;

        if (hasPixelChangeData || hasSemanticChangeData) {

          data.change_analysis = {
            statistics: hasPixelChangeData
              ? {
                  total_pixels: Number(
                    data.total_pixels || 0
                  ),

                  changed_pixels: Number(
                    data.changed_pixels || 0
                  ),

                  changed_percentage: Number(
                    data.changed_area_percent || 0
                  ),

                  unchanged_percentage: Number(
                    data.unchanged_area_percent || 0
                  ),
                }
              : undefined,

            change_map:
              data.change_map || undefined,

            image_size: {
              width: Number(
                data.image?.width || 0
              ),

              height: Number(
                data.image?.height || 0
              ),
            },

            before_evidence: beforeEvidence,
            after_evidence: afterEvidence,
            changes: semanticChanges,
          };
        }
      }


      const result:
        AnalysisResult = data;


      setAnalysisResult(result);
      setAnalyzed(true);


    } catch (err) {

      console.error(
        "Analysis error:",
        err
      );


      if (err instanceof Error) {

        setError(
          err.message
        );

      } else {

        setError(
          "Unable to connect to the SatQuery backend. Make sure FastAPI is running on port 8000."
        );
      }

    } finally {

      setAnalyzing(false);
    }
  };


  // ----------------------------------------------------------
  // REMOVE IMAGE
  // ----------------------------------------------------------

  const removeImage = (
    second = false
  ) => {

    if (second) {

      if (secondImage) {
        URL.revokeObjectURL(
          secondImage
        );
      }

      setSecondImage(null);
      setSecondImageFile(null);

    } else {

      if (image) {
        URL.revokeObjectURL(
          image
        );
      }

      setImage(null);
      setImageFile(null);
    }


    setAnalyzed(false);
    setAnalysisResult(null);
    setError(null);
  };


  // ----------------------------------------------------------
  // FORMAT TASK NAME
  // ----------------------------------------------------------

  const formatTaskName = (
    task: string
  ) => {

    return task
      .replace(/_/g, " ")
      .replace(/\b\w/g, char =>
        char.toUpperCase()
      );
  };


  // ----------------------------------------------------------
  // CONFIDENCE DISPLAY
  // ----------------------------------------------------------

  const getConfidenceDisplay = () => {

    if (
      analysisResult?.confidence === null ||
      analysisResult?.confidence === undefined
    ) {

      return "Not calculated";
    }

    return `${(
      analysisResult.confidence * 100
    ).toFixed(1)}%`;
  };


  const getConfidenceLabel = () => {

    if (
      analysisResult?.detected_task ===
      "optical_sar_analysis"
    ) {
      return "Baseline Fusion Score";
    }

    return "Model Confidence";
  };


  // ----------------------------------------------------------
  // EVIDENCE STATUS
  // ----------------------------------------------------------

  const getEvidenceStatus = () => {

    if (analysisResult?.detected_task === "grounding") {

      return analysisResult.grounding_map
        ? "Grounding map generated"
        : "Grounding analysis completed";
    }

    if (
      analysisResult?.change_analysis?.change_map
    ) {

      return "Change map generated";
    }


    if (
      analysisResult?.change_analysis?.before_evidence ||
      analysisResult?.before_evidence
    ) {

      return "Semantic change evidence acquired";
    }


    if (
      analysisResult?.evidence
    ) {

      return "Visual evidence acquired";
    }


    return "Analysis completed";
  };


  // ==========================================================
  // UI
  // ==========================================================

  return (

    <div className="relative min-h-screen overflow-hidden bg-[#030712] text-white">


      {/* =====================================================
          SPACE BACKGROUND
      ====================================================== */}

      <div className="pointer-events-none fixed inset-0 overflow-hidden">

        {/* Deep space gradient */}

        <div
          className="
            absolute inset-0
            bg-[radial-gradient(circle_at_50%_-10%,rgba(14,165,233,0.13),transparent_40%),radial-gradient(circle_at_90%_30%,rgba(59,130,246,0.08),transparent_30%),radial-gradient(circle_at_10%_80%,rgba(6,182,212,0.06),transparent_30%)]
          "
        />


        {/* Stars */}

        {stars.map(star => (

          <span
            key={star.id}
            className="
              absolute
              h-[2px]
              w-[2px]
              rounded-full
              bg-cyan-200/50
              animate-pulse
            "
            style={{
              left: star.left,
              top: star.top,
              animationDelay: star.delay,
              animationDuration: star.duration,
            }}
          />

        ))}


        {/* Atmospheric glow */}

        <div
          className="
            absolute
            left-1/2
            top-[-300px]
            h-[650px]
            w-[650px]
            -translate-x-1/2
            rounded-full
            bg-cyan-500/[0.035]
            blur-3xl
          "
        />


        {/* Grid */}

        <div
          className="absolute inset-0 opacity-[0.025]"
          style={{
            backgroundImage:
              "linear-gradient(rgba(34,211,238,.7) 1px, transparent 1px), linear-gradient(90deg, rgba(34,211,238,.7) 1px, transparent 1px)",
            backgroundSize:
              "60px 60px",
          }}
        />

      </div>


      {/* =====================================================
          HEADER
      ====================================================== */}

      <header
        className="
          sticky
          top-0
          z-50
          border-b
          border-cyan-400/10
          bg-[#030712]/80
          backdrop-blur-2xl
        "
      >

        <div
          className="
            mx-auto
            flex
            max-w-7xl
            items-center
            justify-between
            px-6
            py-4
          "
        >

          <div className="flex items-center gap-3">

            <div
              className="
                relative
                flex
                h-11
                w-11
                items-center
                justify-center
                rounded-xl
                border
                border-cyan-400/30
                bg-cyan-400/10
                shadow-[0_0_25px_rgba(34,211,238,0.12)]
              "
            >

              <Satellite
                className="
                  h-5
                  w-5
                  text-cyan-300
                "
              />

              <span
                className="
                  absolute
                  -right-1
                  -top-1
                  h-2
                  w-2
                  animate-ping
                  rounded-full
                  bg-cyan-300
                "
              />

              <span
                className="
                  absolute
                  -right-1
                  -top-1
                  h-2
                  w-2
                  rounded-full
                  bg-cyan-300
                "
              />

            </div>


            <div>

              <h1
                className="
                  text-lg
                  font-semibold
                  tracking-[0.12em]
                "
              >

                SATQUERY{" "}

                <span className="text-cyan-300">
                  AI
                </span>

              </h1>


              <p
                className="
                  text-[9px]
                  tracking-[0.28em]
                  text-slate-500
                "
              >
                ORBITAL INTELLIGENCE SYSTEM
              </p>

            </div>

          </div>


          {/* System status */}

          <div className="flex items-center gap-3">

            <div
              className="
                hidden
                items-center
                gap-2
                text-[10px]
                uppercase
                tracking-widest
                text-slate-600
                sm:flex
              "
            >

              <Radio className="h-3 w-3" />

              Neural Vision Link

            </div>


            <div
              className="
                flex
                items-center
                gap-2
                rounded-full
                border
                border-emerald-400/20
                bg-emerald-400/[0.04]
                px-3
                py-1.5
              "
            >

              <span className="relative flex h-2 w-2">

                <span
                  className="
                    absolute
                    inline-flex
                    h-full
                    w-full
                    animate-ping
                    rounded-full
                    bg-emerald-400
                    opacity-60
                  "
                />

                <span
                  className="
                    relative
                    inline-flex
                    h-2
                    w-2
                    rounded-full
                    bg-emerald-400
                  "
                />

              </span>


              <span
                className="
                  text-[10px]
                  font-medium
                  tracking-wider
                  text-emerald-300
                "
              >
                SYSTEM ONLINE
              </span>

            </div>

          </div>

        </div>

      </header>


      {/* =====================================================
          MAIN
      ====================================================== */}

      <main
        className="
          relative
          z-10
          mx-auto
          max-w-7xl
          px-6
          py-12
        "
      >


        {/* ===================================================
            HERO
        ==================================================== */}

        <section className="relative mb-12">

          <div
            className="
              mb-4
              flex
              items-center
              gap-2
            "
          >

            <div
              className="
                flex
                items-center
                gap-2
                rounded-full
                border
                border-cyan-400/20
                bg-cyan-400/[0.04]
                px-3
                py-1.5
              "
            >

              <Sparkles
                className="
                  h-3
                  w-3
                  text-cyan-300
                "
              />

              <span
                className="
                  text-[9px]
                  font-medium
                  uppercase
                  tracking-[0.22em]
                  text-cyan-300
                "
              >
                Agentic Satellite Analysis
              </span>

            </div>


            <span
              className="
                hidden
                h-px
                w-20
                bg-gradient-to-r
                from-cyan-400/40
                to-transparent
                sm:block
              "
            />

          </div>


          <h2
            className="
              max-w-4xl
              text-4xl
              font-semibold
              leading-tight
              tracking-tight
              md:text-6xl
            "
          >

            Ask the Earth

            <br />

            <span
              className="
                bg-gradient-to-r
                from-cyan-300
                via-sky-400
                to-blue-500
                bg-clip-text
                text-transparent
              "
            >
              anything.
            </span>

          </h2>


          <p
            className="
              mt-5
              max-w-2xl
              text-sm
              leading-7
              text-slate-400
            "
          >
            SatQuery transforms satellite imagery
            into actionable visual intelligence
            using AI-powered remote-sensing
            analysis.
          </p>


          {/* Indicators */}

          <div className="mt-7 flex flex-wrap gap-3">

            <StatusPill
              icon={
                <Cpu className="h-3 w-3" />
              }
              text="AI VISION ACTIVE"
            />

            <StatusPill
              icon={
                <ScanLine className="h-3 w-3" />
              }
              text="IMAGE ANALYSIS"
            />

            <StatusPill
              icon={
                <Waves className="h-3 w-3" />
              }
              text="REMOTE SENSING"
            />

          </div>

        </section>


        {/* ===================================================
            INPUT AREA
        ==================================================== */}

        <section
          className="
            grid
            gap-5
            lg:grid-cols-2
          "
        >


          {/* =================================================
              IMAGE PANEL
          ================================================== */}

          <div
            className="
              group
              relative
              overflow-hidden
              rounded-2xl
              border
              border-cyan-400/10
              bg-white/[0.025]
              p-5
              shadow-[0_20px_60px_rgba(0,0,0,0.25)]
              backdrop-blur-xl
            "
          >

            <div
              className="
                absolute
                left-0
                right-0
                top-0
                h-px
                bg-gradient-to-r
                from-transparent
                via-cyan-400/50
                to-transparent
              "
            />


            <div
              className="
                mb-5
                flex
                items-center
                justify-between
              "
            >

              <div>

                <div
                  className="
                    flex
                    items-center
                    gap-2
                  "
                >

                  <span
                    className="
                      h-1.5
                      w-1.5
                      rounded-full
                      bg-cyan-400
                      shadow-[0_0_10px_rgba(34,211,238,.8)]
                    "
                  />

                  <p
                    className="
                      text-[10px]
                      font-medium
                      uppercase
                      tracking-[0.18em]
                      text-cyan-300
                    "
                  >
                    Imagery Input
                  </p>

                </div>


                <p
                  className="
                    mt-1
                    text-xs
                    text-slate-600
                  "
                >
                  Optical · Multispectral · SAR
                </p>

              </div>


              <ImageIcon
                className="
                  h-5
                  w-5
                  text-slate-700
                "
              />

            </div>


            <div
              className="
                grid
                gap-4
                sm:grid-cols-2
              "
            >

              {/* PRIMARY */}

              <ImageDropZone
                id="image-upload"
                image={image}
                label="Primary imagery"
                emptyTitle="Upload satellite image"
                emptySubtitle="PNG · JPG · TIFF"
                icon={
                  <Upload className="h-5 w-5" />
                }
                onChange={(e) =>
                  handleImageUpload(e)
                }
                onRemove={() =>
                  removeImage()
                }
              />


              {/* SECONDARY */}

              <ImageDropZone
                id="second-image-upload"
                image={secondImage}
                label="Secondary imagery"
                emptyTitle="Optional second image"
                emptySubtitle="Change / temporal analysis"
                icon={
                  <FileImage className="h-5 w-5" />
                }
                onChange={(e) =>
                  handleImageUpload(e, true)
                }
                onRemove={() =>
                  removeImage(true)
                }
              />

            </div>


            <div
              className="
                mt-4
                flex
                items-center
                justify-between
                text-[9px]
                uppercase
                tracking-widest
                text-slate-700
              "
            >

              <span>
                Input channel 01
              </span>

              <span>
                {imageFile
                  ? "SIGNAL ACQUIRED"
                  : "AWAITING SIGNAL"}
              </span>

            </div>

          </div>


          {/* =================================================
              QUERY PANEL
          ================================================== */}

          <div
            className="
              relative
              overflow-hidden
              rounded-2xl
              border
              border-cyan-400/10
              bg-white/[0.025]
              p-5
              shadow-[0_20px_60px_rgba(0,0,0,0.25)]
              backdrop-blur-xl
            "
          >

            <div
              className="
                absolute
                right-0
                top-0
                h-32
                w-32
                rounded-full
                bg-cyan-400/[0.04]
                blur-3xl
              "
            />


            <div className="mb-5">

              <div
                className="
                  flex
                  items-center
                  gap-2
                "
              >

                <span
                  className="
                    h-1.5
                    w-1.5
                    rounded-full
                    bg-cyan-400
                    shadow-[0_0_10px_rgba(34,211,238,.8)]
                  "
                />

                <p
                  className="
                    text-[10px]
                    font-medium
                    uppercase
                    tracking-[0.18em]
                    text-cyan-300
                  "
                >
                  Intelligence Query
                </p>

              </div>


              <p
                className="
                  mt-1
                  text-xs
                  text-slate-600
                "
              >
                Communicate with the satellite
                vision system
              </p>

            </div>


            <div className="relative">

              <textarea
                value={query}
                onChange={(e) => {

                  setQuery(
                    e.target.value
                  );

                  setAnalyzed(false);
                  setAnalysisResult(null);
                  setError(null);
                }}
                placeholder="Ask SatQuery what you want to know..."
                className="
                  h-[148px]
                  w-full
                  resize-none
                  rounded-xl
                  border
                  border-white/10
                  bg-[#040a13]/80
                  p-4
                  text-sm
                  leading-6
                  text-slate-200
                  outline-none
                  transition
                  placeholder:text-slate-700
                  focus:border-cyan-400/40
                  focus:shadow-[0_0_25px_rgba(34,211,238,0.05)]
                "
              />


              <div
                className="
                  pointer-events-none
                  absolute
                  bottom-3
                  right-3
                  text-[9px]
                  uppercase
                  tracking-widest
                  text-slate-700
                "
              >
                Natural Language
              </div>

            </div>


            {/* Suggestions */}

            <div
              className="
                mt-4
                flex
                flex-wrap
                gap-2
              "
            >

              {[
                "Describe this image",
                "Highlight water bodies",
                "What changed?",
                "Analyze land cover",
              ].map(suggestion => (

                <button
                  key={suggestion}
                  onClick={() => {

                    setQuery(
                      suggestion
                    );

                    setAnalyzed(false);
                    setAnalysisResult(null);
                    setError(null);
                  }}
                  className="
                    rounded-full
                    border
                    border-white/10
                    bg-white/[0.015]
                    px-3
                    py-1.5
                    text-[10px]
                    text-slate-500
                    transition-all
                    hover:border-cyan-400/30
                    hover:bg-cyan-400/[0.04]
                    hover:text-cyan-300
                  "
                >
                  {suggestion}
                </button>

              ))}

            </div>

          </div>

        </section>


        {/* ===================================================
            ERROR
        ==================================================== */}

        {error && (

          <div
            className="
              mt-5
              rounded-xl
              border
              border-red-400/20
              bg-red-400/[0.04]
              px-4
              py-3
              text-sm
              text-red-300
            "
          >
            {error}
          </div>

        )}


        {/* ===================================================
            ANALYZE BUTTON
        ==================================================== */}

        <div
          className="
            mt-6
            flex
            justify-end
          "
        >

          <button
            onClick={handleAnalyze}
            disabled={
              !imageFile ||
              !query.trim() ||
              analyzing
            }
            className="
              group
              relative
              flex
              items-center
              gap-3
              overflow-hidden
              rounded-xl
              border
              border-cyan-300/30
              bg-cyan-400
              px-7
              py-3.5
              text-xs
              font-bold
              uppercase
              tracking-wider
              text-slate-950
              shadow-[0_0_30px_rgba(34,211,238,0.15)]
              transition-all
              hover:bg-cyan-300
              hover:shadow-[0_0_40px_rgba(34,211,238,0.25)]
              disabled:cursor-not-allowed
              disabled:opacity-30
            "
          >

            {analyzing && (

              <span
                className="
                  absolute
                  inset-0
                  animate-pulse
                  bg-white/20
                "
              />

            )}


            {analyzing ? (

              <>
                <ScanLine
                  className="
                    relative
                    h-4
                    w-4
                    animate-pulse
                  "
                />

                <span className="relative">
                  Scanning imagery...
                </span>
              </>

            ) : (

              <>
                <span>
                  Initiate Analysis
                </span>

                <ArrowRight
                  className="
                    h-4
                    w-4
                    transition-transform
                    group-hover:translate-x-1
                  "
                />
              </>

            )}

          </button>

        </div>


        {/* ===================================================
            LOADING
        ==================================================== */}

        {analyzing && (

          <div
            className="
              mt-10
              overflow-hidden
              rounded-2xl
              border
              border-cyan-400/10
              bg-cyan-400/[0.02]
            "
          >

            <div
              className="
                relative
                h-1
                overflow-hidden
                bg-cyan-400/5
              "
            >

              <div
                className="
                  absolute
                  h-full
                  w-1/3
                  animate-[scan_1.5s_ease-in-out_infinite]
                  bg-gradient-to-r
                  from-transparent
                  via-cyan-400
                  to-transparent
                "
              />

            </div>


            <div
              className="
                flex
                items-center
                gap-4
                p-5
              "
            >

              <div
                className="
                  flex
                  h-10
                  w-10
                  items-center
                  justify-center
                  rounded-xl
                  border
                  border-cyan-400/20
                  bg-cyan-400/5
                "
              >

                <Satellite
                  className="
                    h-5
                    w-5
                    animate-pulse
                    text-cyan-300
                  "
                />

              </div>


              <div>

                <p
                  className="
                    text-xs
                    font-medium
                    uppercase
                    tracking-wider
                    text-cyan-300
                  "
                >
                  Neural analysis in progress
                </p>

                <p
                  className="
                    mt-1
                    text-[11px]
                    text-slate-600
                  "
                >
                  Agent controller is selecting
                  the appropriate analysis pipeline
                </p>

              </div>

            </div>

          </div>

        )}


        {/* ===================================================
            RESULTS
        ==================================================== */}

        {analyzed &&
          analysisResult && (

            <section
              className="
                mt-14
                animate-[fadeUp_.6s_ease-out]
              "
            >

              {/* RESULT HEADER */}

              <div
                className="
                  mb-6
                  flex
                  items-end
                  justify-between
                "
              >

                <div>

                  <div
                    className="
                      flex
                      items-center
                      gap-2
                    "
                  >

                    <span
                      className="
                        h-1.5
                        w-1.5
                        rounded-full
                        bg-emerald-400
                        shadow-[0_0_10px_rgba(52,211,153,.8)]
                      "
                    />

                    <p
                      className="
                        text-[10px]
                        font-medium
                        uppercase
                        tracking-[0.2em]
                        text-emerald-300
                      "
                    >
                      Intelligence Report
                    </p>

                  </div>


                  <h3
                    className="
                      mt-2
                      text-2xl
                      font-semibold
                      tracking-tight
                    "
                  >
                    Analysis complete
                  </h3>

                </div>


                <div
                  className="
                    hidden
                    items-center
                    gap-2
                    rounded-full
                    border
                    border-emerald-400/20
                    bg-emerald-400/[0.04]
                    px-3
                    py-1.5
                    text-[10px]
                    uppercase
                    tracking-wider
                    text-emerald-300
                    sm:flex
                  "
                >

                  <CheckCircle2
                    className="h-3.5 w-3.5"
                  />

                  ANALYSIS COMPLETE

                </div>

              </div>


              {/* =================================================
                  METRICS
              ================================================== */}

              <div
                className="
                  grid
                  gap-3
                  md:grid-cols-3
                "
              >

                <ResultCard
                  icon={
                    <BrainCircuit className="h-4 w-4" />
                  }
                  label="Detected Task"
                  value={formatTaskName(
                    analysisResult.detected_task
                  )}
                />


                <ResultCard
                  icon={
                    <Activity className="h-4 w-4" />
                  }
                  label={getConfidenceLabel()}
                  value={
                    getConfidenceDisplay()
                  }
                />


                <ResultCard
                  icon={
                    <Map className="h-4 w-4" />
                  }
                  label="Evidence Status"
                  value={
                    getEvidenceStatus()
                  }
                />

              </div>


              {/* =================================================
                  AGENT DECISION
              ================================================== */}

              {analysisResult.agent && (

                <div
                  className="
                    mt-5
                    rounded-2xl
                    border
                    border-cyan-400/10
                    bg-cyan-400/[0.02]
                    p-5
                  "
                >

                  <div
                    className="
                      flex
                      items-center
                      gap-2
                    "
                  >

                    <GitCompare
                      className="
                        h-4
                        w-4
                        text-cyan-300
                      "
                    />

                    <p
                      className="
                        text-[10px]
                        uppercase
                        tracking-[0.18em]
                        text-slate-500
                      "
                    >
                      Agent Controller Decision
                    </p>

                  </div>


                  <div
                    className="
                      mt-4
                      grid
                      gap-4
                      md:grid-cols-2
                    "
                  >

                    <div>

                      <p
                        className="
                          text-[9px]
                          uppercase
                          tracking-widest
                          text-slate-600
                        "
                      >
                        Selected task
                      </p>

                      <p
                        className="
                          mt-1
                          text-sm
                          font-medium
                          text-cyan-300
                        "
                      >
                        {formatTaskName(
                          analysisResult.agent.task
                        )}
                      </p>

                    </div>


                    <div>

                      <p
                        className="
                          text-[9px]
                          uppercase
                          tracking-widest
                          text-slate-600
                        "
                      >
                        Reason
                      </p>

                      <p
                        className="
                          mt-1
                          text-xs
                          leading-6
                          text-slate-400
                        "
                      >
                        {analysisResult.agent.reason}
                      </p>

                    </div>

                  </div>

                </div>

              )}


              {/* =================================================
                  CHANGE ANALYSIS
              ================================================== */}

              {analysisResult.change_analysis && (

                <section className="mt-5">

                  <div
                    className="
                      relative
                      overflow-hidden
                      rounded-2xl
                      border
                      border-cyan-400/10
                      bg-[#040a12]
                      shadow-[0_20px_70px_rgba(0,0,0,0.3)]
                    "
                  >

                    {/* Header */}

                    <div
                      className="
                        flex
                        items-center
                        justify-between
                        border-b
                        border-white/10
                        px-5
                        py-3
                      "
                    >

                      <div
                        className="
                          flex
                          items-center
                          gap-2
                        "
                      >

                        <GitCompare
                          className="
                            h-3.5
                            w-3.5
                            text-cyan-300
                          "
                        />

                        <span
                          className="
                            text-[10px]
                            uppercase
                            tracking-[0.16em]
                            text-slate-500
                          "
                        >
                          Change Detection Map
                        </span>

                      </div>


                      <span
                        className="
                          text-[9px]
                          uppercase
                          tracking-widest
                          text-cyan-400
                        "
                      >
                        Pixel Difference Analysis
                      </span>

                    </div>


                    {/* Change map */}

                    {analysisResult.change_analysis.change_map && (

                    <div
                      className="
                        relative
                        flex
                        min-h-[350px]
                        items-center
                        justify-center
                        overflow-hidden
                        bg-[#02060c]
                      "
                    >

                      <img
                        src={
                          analysisResult
                            .change_analysis
                            .change_map
                        }
                        alt="Detected visual changes"
                        className="
                          max-h-[600px]
                          w-full
                          object-contain
                        "
                      />


                      {/* Scan line */}

                      <div
                        className="
                          pointer-events-none
                          absolute
                          inset-x-0
                          top-0
                          h-px
                          animate-[scanVertical_4s_linear_infinite]
                          bg-cyan-300/70
                          shadow-[0_0_20px_rgba(34,211,238,.8)]
                        "
                      />


                      {/* Feed label */}

                      <div
                        className="
                          absolute
                          left-4
                          top-4
                          rounded
                          border
                          border-cyan-400/20
                          bg-[#03101a]/80
                          px-2
                          py-1
                          font-mono
                          text-[9px]
                          text-cyan-300
                        "
                      >
                        CHANGE FEED // 02
                      </div>


                      <div
                        className="
                          absolute
                          right-4
                          top-4
                          font-mono
                          text-[9px]
                          text-slate-500
                        "
                      >
                        SATQUERY-DIFF
                      </div>

                    </div>

                    )}

                  </div>


                  {/* CHANGE STATISTICS */}

                  {analysisResult.change_analysis.statistics && (

                  <div
                    className="
                      mt-4
                      grid
                      gap-3
                      md:grid-cols-3
                    "
                  >

                    <ChangeStatCard
                      label="Changed Area"
                      value={`${analysisResult.change_analysis.statistics.changed_percentage.toFixed(2)}%`}
                      description="Detected visual difference"
                    />


                    <ChangeStatCard
                      label="Unchanged Area"
                      value={`${analysisResult.change_analysis.statistics.unchanged_percentage.toFixed(2)}%`}
                      description="Visually stable region"
                    />


                    <ChangeStatCard
                      label="Pixels Changed"
                      value={analysisResult.change_analysis.statistics.changed_pixels.toLocaleString()}
                      description={`of ${analysisResult.change_analysis.statistics.total_pixels.toLocaleString()} pixels`}
                    />

                  </div>

                  )}


                  {/* =================================================
                      SEMANTIC CHANGE EVIDENCE
                  ================================================== */}

                  {analysisResult.change_analysis &&
                    (
                      analysisResult.change_analysis.before_evidence ||
                      analysisResult.change_analysis.after_evidence ||
                      analysisResult.change_analysis.changes
                    ) && (

                    <div className="mt-5">

                      <div
                        className="
                          rounded-2xl
                          border
                          border-cyan-400/10
                          bg-white/[0.02]
                          p-5
                        "
                      >

                        <div className="flex items-center gap-2">

                          <GitCompare
                            className="h-4 w-4 text-cyan-300"
                          />

                          <p
                            className="
                              text-[10px]
                              uppercase
                              tracking-[0.18em]
                              text-slate-500
                            "
                          >
                            Semantic Change Evidence
                          </p>

                        </div>

                        <p className="mt-1 text-[11px] text-slate-600">
                          AI-assisted comparison of visual features between the two observations
                        </p>

                        <div className="mt-5 overflow-x-auto">

                          <table className="w-full min-w-[600px] text-left">

                            <thead>
                              <tr className="border-b border-white/10">

                                <th className="px-3 py-3 text-[9px] uppercase tracking-widest text-slate-600">
                                  Feature
                                </th>

                                <th className="px-3 py-3 text-[9px] uppercase tracking-widest text-slate-600">
                                  Before
                                </th>

                                <th className="px-3 py-3 text-[9px] uppercase tracking-widest text-slate-600">
                                  After
                                </th>

                                <th className="px-3 py-3 text-[9px] uppercase tracking-widest text-slate-600">
                                  Change
                                </th>

                              </tr>
                            </thead>

                            <tbody>

                              {Object.keys(
                                analysisResult.change_analysis.changes &&
                                Object.keys(analysisResult.change_analysis.changes).length > 0
                                  ? analysisResult.change_analysis.changes
                                  : analysisResult.change_analysis.before_evidence || {}
                              ).map((feature) => {

                                const semanticChange =
                                  analysisResult.change_analysis?.changes?.[feature];

                                const before =
                                  semanticChange?.before ||
                                  analysisResult.change_analysis?.before_evidence?.[feature] ||
                                  "unknown";

                                const after =
                                  semanticChange?.after ||
                                  analysisResult.change_analysis?.after_evidence?.[feature] ||
                                  "unknown";

                                const change =
                                  semanticChange?.change ||
                                  (before === after ? "stable" : "changed");

                                return (

                                  <tr
                                    key={feature}
                                    className="border-b border-white/[0.05]"
                                  >

                                    <td className="px-3 py-3 text-xs font-medium capitalize text-slate-300">
                                      {feature.replace(/_/g, " ")}
                                    </td>

                                    <td className="px-3 py-3">
                                      <EvidenceValueBadge value={before} />
                                    </td>

                                    <td className="px-3 py-3">
                                      <EvidenceValueBadge value={after} />
                                    </td>

                                    <td className="px-3 py-3">
                                      <span
                                        className={
                                          change === "stable"
                                            ? "text-[10px] font-medium uppercase tracking-wider text-slate-500"
                                            : "text-[10px] font-medium uppercase tracking-wider text-cyan-300"
                                        }
                                      >
                                        {change}
                                      </span>
                                    </td>

                                  </tr>
                                );
                              })}

                            </tbody>

                          </table>

                        </div>

                      </div>

                    </div>
                  )}


                  {/* Technical note */}

                  <div
                    className="
                      mt-4
                      rounded-xl
                      border
                      border-cyan-400/10
                      bg-cyan-400/[0.02]
                      px-4
                      py-3
                    "
                  >

                    <p
                      className="
                        text-[10px]
                        leading-5
                        text-slate-500
                      "
                    >
                      <span className="text-cyan-300">
                        Analysis note:
                      </span>{" "}
                      Semantic evidence is generated using
                      the VQA model, while the change map
                      represents pixel-level visual
                      differences between the two observations.
                      Geospatial image registration will be
                      added to reduce false changes caused by
                      image misalignment.
                    </p>

                  </div>

                </section>

              )}


              {/* =================================================
                  GROUNDING ANALYSIS
              ================================================== */}

              {analysisResult.detected_task === "grounding" &&
                analysisResult.grounding && (

                <section className="mt-5">

                  <div
                    className="
                      relative
                      overflow-hidden
                      rounded-2xl
                      border
                      border-cyan-400/10
                      bg-[#040a12]
                      shadow-[0_20px_70px_rgba(0,0,0,0.3)]
                    "
                  >

                    {/* Header */}

                    <div
                      className="
                        flex
                        items-center
                        justify-between
                        border-b
                        border-white/10
                        px-5
                        py-3
                      "
                    >

                      <div className="flex items-center gap-2">

                        <ScanLine
                          className="h-3.5 w-3.5 text-cyan-300"
                        />

                        <span
                          className="
                            text-[10px]
                            uppercase
                            tracking-[0.16em]
                            text-slate-500
                          "
                        >
                          Grounding Analysis
                        </span>

                      </div>

                      <span
                        className="
                          text-[9px]
                          uppercase
                          tracking-widest
                          text-cyan-400
                        "
                      >
                        Candidate Regions
                      </span>

                    </div>


                    {/* Grounding map */}

                    <div
                      className="
                        relative
                        flex
                        min-h-[350px]
                        items-center
                        justify-center
                        overflow-hidden
                        bg-[#02060c]
                      "
                    >

                      <img
                        src={analysisResult.grounding.grounding_map}
                        alt="Grounding detections"
                        className="
                          max-h-[600px]
                          w-full
                          object-contain
                        "
                      />

                      <div
                        className="
                          pointer-events-none
                          absolute
                          inset-x-0
                          top-0
                          h-px
                          animate-[scanVertical_4s_linear_infinite]
                          bg-cyan-300/70
                          shadow-[0_0_20px_rgba(34,211,238,.8)]
                        "
                      />

                      <div
                        className="
                          absolute
                          left-4
                          top-4
                          rounded
                          border
                          border-cyan-400/20
                          bg-[#03101a]/80
                          px-2
                          py-1
                          font-mono
                          text-[9px]
                          text-cyan-300
                        "
                      >
                        GROUNDING FEED // 03
                      </div>

                      <div
                        className="
                          absolute
                          right-4
                          top-4
                          font-mono
                          text-[9px]
                          text-slate-500
                        "
                      >
                        ASTRA-GROUND
                      </div>

                    </div>

                  </div>


                  {/* Grounding metrics */}

                  <div
                    className="
                      mt-4
                      grid
                      gap-3
                      md:grid-cols-3
                    "
                  >

                    <ChangeStatCard
                      label="Candidate Regions"
                      value={analysisResult.grounding.detection_count.toString()}
                      description="After confidence + NMS filtering"
                    />

                    <ChangeStatCard
                      label="Average Confidence"
                      value={`${((analysisResult.grounding.average_confidence || 0) * 100).toFixed(1)}%`}
                      description="Across retained detections"
                    />

                    <ChangeStatCard
                      label="Detection Threshold"
                      value={`${((analysisResult.grounding.parameters?.detection_threshold || 0) * 100).toFixed(0)}%`}
                      description="OWLv2 confidence filter"
                    />

                  </div>


                  {/* Concepts searched */}

                  <div
                    className="
                      mt-4
                      rounded-xl
                      border
                      border-white/10
                      bg-white/[0.02]
                      px-4
                      py-3
                    "
                  >

                    <p
                      className="
                        text-[9px]
                        uppercase
                        tracking-[0.16em]
                        text-slate-600
                      "
                    >
                      Concepts searched
                    </p>

                    <div className="mt-2 flex flex-wrap gap-2">

                      {analysisResult.grounding.labels_searched.map(
                        (label) => (

                        <span
                          key={label}
                          className="
                            rounded-full
                            border
                            border-cyan-400/10
                            bg-cyan-400/[0.03]
                            px-3
                            py-1.5
                            text-[9px]
                            uppercase
                            tracking-wider
                            text-cyan-300
                          "
                        >
                          {label}
                        </span>

                      ))}

                    </div>

                  </div>


                  {/* Detection table */}

                  {analysisResult.grounding.detections.length > 0 && (

                    <div
                      className="
                        mt-4
                        rounded-xl
                        border
                        border-white/10
                        bg-white/[0.02]
                        p-5
                      "
                    >

                      <div className="flex items-center gap-2">

                        <Building2
                          className="h-4 w-4 text-cyan-300"
                        />

                        <p
                          className="
                            text-[10px]
                            uppercase
                            tracking-[0.18em]
                            text-slate-500
                          "
                        >
                          Detected Candidate Regions
                        </p>

                      </div>

                      <div className="mt-4 overflow-x-auto">

                        <table className="w-full min-w-[600px] text-left">

                          <thead>
                            <tr className="border-b border-white/10">
                              <th className="px-3 py-3 text-[9px] uppercase tracking-widest text-slate-600">
                                #
                              </th>
                              <th className="px-3 py-3 text-[9px] uppercase tracking-widest text-slate-600">
                                Concept
                              </th>
                              <th className="px-3 py-3 text-[9px] uppercase tracking-widest text-slate-600">
                                Confidence
                              </th>
                              <th className="px-3 py-3 text-[9px] uppercase tracking-widest text-slate-600">
                                Bounding Box
                              </th>
                            </tr>
                          </thead>

                          <tbody>

                            {analysisResult.grounding.detections.map(
                              (detection, index) => (

                              <tr
                                key={`${detection.label}-${index}`}
                                className="border-b border-white/[0.05]"
                              >

                                <td className="px-3 py-3 text-xs text-slate-500">
                                  {index + 1}
                                </td>

                                <td className="px-3 py-3">
                                  <span
                                    className="
                                      text-xs
                                      font-medium
                                      capitalize
                                      text-slate-300
                                    "
                                  >
                                    {detection.label}
                                  </span>
                                </td>

                                <td className="px-3 py-3">
                                  <span
                                    className="
                                      rounded-full
                                      border
                                      border-cyan-400/10
                                      bg-cyan-400/[0.03]
                                      px-2.5
                                      py-1
                                      text-[9px]
                                      font-medium
                                      tracking-wider
                                      text-cyan-300
                                    "
                                  >
                                    {(detection.confidence * 100).toFixed(1)}%
                                  </span>
                                </td>

                                <td className="px-3 py-3 font-mono text-[9px] text-slate-500">
                                  ({detection.box.x1.toFixed(0)}, {detection.box.y1.toFixed(0)}) → ({detection.box.x2.toFixed(0)}, {detection.box.y2.toFixed(0)})
                                </td>

                              </tr>

                            ))}

                          </tbody>

                        </table>

                      </div>

                    </div>

                  )}


                  {/* Grounding note */}

                  <div
                    className="
                      mt-4
                      rounded-xl
                      border
                      border-amber-400/10
                      bg-amber-400/[0.025]
                      px-4
                      py-3
                    "
                  >

                    <p
                      className="
                        text-[10px]
                        leading-5
                        text-slate-500
                      "
                    >
                      <span className="text-amber-300">
                        Grounding note:
                      </span>{" "}
                      These are candidate regions produced by a
                      general zero-shot grounding model.
                      Confidence filtering and non-maximum
                      suppression are applied before the
                      detections are shown. Remote-sensing
                      specific adaptation is planned for a
                      later model-improvement stage.
                    </p>

                  </div>

                </section>

              )}


              {/* =================================================
                  OPTICAL + SAR ANALYSIS
              ================================================== */}

              {analysisResult.detected_task === "optical_sar_analysis" &&
                analysisResult.optical_sar && (

                <section className="mt-5">

                  <div
                    className="
                      relative
                      overflow-hidden
                      rounded-2xl
                      border
                      border-cyan-400/10
                      bg-[#040a12]
                      shadow-[0_20px_70px_rgba(0,0,0,0.3)]
                    "
                  >

                    {/* Header */}

                    <div
                      className="
                        flex
                        items-center
                        justify-between
                        border-b
                        border-white/10
                        px-5
                        py-3
                      "
                    >

                      <div className="flex items-center gap-2">

                        <Waves
                          className="h-3.5 w-3.5 text-cyan-300"
                        />

                        <span
                          className="
                            text-[10px]
                            uppercase
                            tracking-[0.16em]
                            text-slate-500
                          "
                        >
                          Optical + SAR Analysis
                        </span>

                      </div>

                      <span
                        className="
                          text-[9px]
                          uppercase
                          tracking-widest
                          text-cyan-400
                        "
                      >
                        Cross-Modal Feed
                      </span>

                    </div>


                    {/* Input pair */}

                    <div
                      className="
                        grid
                        gap-4
                        border-b
                        border-white/10
                        p-5
                        md:grid-cols-2
                      "
                    >

                      {[
                        {
                          src: image,
                          label: "OPTICAL INPUT",
                          subtitle: "Sentinel-2 / RGB",
                        },
                        {
                          src: secondImage,
                          label: "SAR INPUT",
                          subtitle: "Sentinel-1 / Radar",
                        },
                      ].map((item) => (

                        <div
                          key={item.label}
                          className="
                            overflow-hidden
                            rounded-xl
                            border
                            border-white/10
                            bg-black/20
                          "
                        >

                          <div
                            className="
                              flex
                              items-center
                              justify-between
                              border-b
                              border-white/[0.06]
                              px-3
                              py-2
                            "
                          >

                            <div>
                              <p
                                className="
                                  text-[9px]
                                  font-medium
                                  tracking-[0.18em]
                                  text-cyan-300
                                "
                              >
                                {item.label}
                              </p>

                              <p
                                className="
                                  mt-1
                                  text-[8px]
                                  tracking-wider
                                  text-slate-600
                                "
                              >
                                {item.subtitle}
                              </p>
                            </div>

                            <Cpu className="h-3.5 w-3.5 text-slate-600" />

                          </div>

                          {item.src ? (
                            <img
                              src={item.src}
                              alt={item.label}
                              className="h-52 w-full object-cover"
                            />
                          ) : (
                            <div className="flex h-52 items-center justify-center text-[10px] text-slate-600">
                              Input unavailable
                            </div>
                          )}

                        </div>

                      ))}

                    </div>


                    {/* Metrics */}

                    <div
                      className="
                        grid
                        gap-3
                        p-5
                        md:grid-cols-4
                      "
                    >

                      <ChangeStatCard
                        label="Baseline Fusion Score"
                        value={`${((analysisResult.optical_sar.confidence || 0) * 100).toFixed(1)}%`}
                        description="Heuristic cross-modal indicator"
                      />

                      <ChangeStatCard
                        label="Registration"
                        value={
                          formatTaskName(
                            analysisResult.optical_sar.registration?.status ||
                            "unknown"
                          )
                        }
                        description="ORB + RANSAC alignment check"
                      />

                      <ChangeStatCard
                        label="Optical Size"
                        value={
                          analysisResult.optical_sar.image_size
                            ? `${analysisResult.optical_sar.image_size.width} × ${analysisResult.optical_sar.image_size.height}`
                            : "—"
                        }
                        description="Common analysis canvas"
                      />

                      <ChangeStatCard
                        label="Matched Features"
                        value={
                          (
                            analysisResult.optical_sar.registration?.inliers ||
                            0
                          ).toString()
                        }
                        description={`${analysisResult.optical_sar.registration?.good_matches || 0} candidate matches`}
                      />

                    </div>


                    {/* Registration / reliability warning */}

                    {(
                      analysisResult.optical_sar.reliability === "caution" ||
                      (
                        analysisResult.optical_sar.registration?.status &&
                        analysisResult.optical_sar.registration.status !== "aligned_candidate"
                      )
                    ) && (
                      <div
                        className="
                          mx-5
                          mb-4
                          rounded-xl
                          border
                          border-amber-400/15
                          bg-amber-400/[0.035]
                          px-4
                          py-3
                        "
                      >
                        <p className="text-[10px] leading-5 text-slate-500">
                          <span className="text-amber-300">
                            Alignment caution:
                          </span>{" "}
                          The optical and SAR images could not be strongly verified as
                          spatially aligned using ORB/RANSAC. Cross-modal findings are
                          therefore prototype indicators and should be interpreted with caution.
                        </p>
                      </div>
                    )}

                    {/* Optical / SAR indicators */}

                    <div
                      className="
                        grid
                        gap-4
                        px-5
                        pb-5
                        md:grid-cols-2
                      "
                    >

                      <div
                        className="
                          rounded-xl
                          border
                          border-white/10
                          bg-white/[0.02]
                          p-5
                        "
                      >

                        <div className="flex items-center gap-2">
                          <TreePine className="h-4 w-4 text-cyan-300" />
                          <p className="text-[10px] uppercase tracking-[0.18em] text-slate-500">
                            Optical Evidence
                          </p>
                        </div>

                        <div className="mt-4 grid grid-cols-3 gap-3">
                          {[
                            ["Vegetation", analysisResult.optical_sar.optical?.vegetation_proxy],
                            ["Water", analysisResult.optical_sar.optical?.water_proxy],
                            ["Built-up", analysisResult.optical_sar.optical?.builtup_proxy],
                          ].map(([label, value]) => (
                            <div key={label as string} className="rounded-lg border border-white/[0.06] bg-black/10 p-3">
                              <p className="text-[9px] uppercase tracking-wider text-slate-600">{label as string}</p>
                              <p className="mt-2 text-xl font-medium text-cyan-300">
                                {value === undefined ? "—" : `${((value as number) * 100).toFixed(1)}%`}
                              </p>
                            </div>
                          ))}
                        </div>

                      </div>


                      <div
                        className="
                          rounded-xl
                          border
                          border-white/10
                          bg-white/[0.02]
                          p-5
                        "
                      >

                        <div className="flex items-center gap-2">
                          <Radio className="h-4 w-4 text-cyan-300" />
                          <p className="text-[10px] uppercase tracking-[0.18em] text-slate-500">
                            SAR Evidence
                          </p>
                        </div>

                        <div className="mt-4 grid grid-cols-3 gap-3">
                          {[
                            ["Intensity", analysisResult.optical_sar.sar?.mean_intensity],
                            ["Texture", analysisResult.optical_sar.sar?.texture_indicator],
                            ["Roughness", analysisResult.optical_sar.sar?.rough_surface_indicator],
                          ].map(([label, value]) => (
                            <div key={label as string} className="rounded-lg border border-white/[0.06] bg-black/10 p-3">
                              <p className="text-[9px] uppercase tracking-wider text-slate-600">{label as string}</p>
                              <p className="mt-2 text-xl font-medium text-cyan-300">
                                {value === undefined ? "—" : `${((value as number) * 100).toFixed(1)}%`}
                              </p>
                            </div>
                          ))}
                        </div>

                      </div>

                    </div>


                    {/* Cross-modal findings */}

                    <div className="px-5 pb-5">

                      <div
                        className="
                          rounded-xl
                          border
                          border-cyan-400/10
                          bg-cyan-400/[0.02]
                          p-5
                        "
                      >

                        <div className="flex items-center gap-2">
                          <GitCompare className="h-4 w-4 text-cyan-300" />
                          <p className="text-[10px] uppercase tracking-[0.18em] text-slate-500">
                            Cross-Modal Findings
                          </p>
                        </div>

                        <div className="mt-4 grid gap-3 md:grid-cols-3">

                          {(analysisResult.optical_sar.cross_modal_findings || []).map((finding) => (

                            <div
                              key={finding.feature}
                              className="rounded-xl border border-white/[0.07] bg-black/10 p-4"
                            >

                              <div className="flex items-center justify-between gap-3">
                                <p className="text-xs font-medium capitalize text-slate-300">
                                  {finding.feature.replace(/_/g, " ")}
                                </p>

                                <span className="rounded-full border border-cyan-400/10 bg-cyan-400/[0.04] px-2.5 py-1 text-[9px] font-medium uppercase tracking-wider text-cyan-300">
                                  {finding.level}
                                </span>
                              </div>

                              <p className="mt-3 text-2xl font-medium text-cyan-300">
                                {(finding.fused_score * 100).toFixed(1)}%
                              </p>

                              {finding.explanation && (
                                <p className="mt-2 text-[9px] leading-5 text-slate-600">
                                  {finding.explanation}
                                </p>
                              )}

                            </div>

                          ))}

                        </div>

                      </div>

                    </div>


                    {/* Registration details */}

                    <div className="px-5 pb-5">
                      <div className="rounded-xl border border-white/10 bg-white/[0.02] p-5">
                        <div className="flex items-center justify-between">
                          <p className="text-[10px] uppercase tracking-[0.18em] text-slate-500">
                            Registration Evidence
                          </p>
                          <span className="font-mono text-[9px] text-slate-600">
                            ORB / RANSAC
                          </span>
                        </div>

                        <div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-5">
                          <ChangeStatCard label="Optical Keypoints" value={(analysisResult.optical_sar.registration?.optical_keypoints || 0).toString()} description="ORB features" />
                          <ChangeStatCard label="SAR Keypoints" value={(analysisResult.optical_sar.registration?.sar_keypoints || 0).toString()} description="ORB features" />
                          <ChangeStatCard label="Good Matches" value={(analysisResult.optical_sar.registration?.good_matches || 0).toString()} description="Lowe ratio matches" />
                          <ChangeStatCard label="RANSAC Inliers" value={(analysisResult.optical_sar.registration?.inliers || 0).toString()} description="Geometric inliers" />
                          <ChangeStatCard label="Inlier Ratio" value={`${((analysisResult.optical_sar.registration?.inlier_ratio || 0) * 100).toFixed(1)}%`} description="RANSAC / good matches" />
                        </div>
                      </div>
                    </div>


                    {/* Technical note */}

                    <div
                      className="
                        mx-5
                        mb-5
                        rounded-xl
                        border
                        border-amber-400/10
                        bg-amber-400/[0.025]
                        px-4
                        py-3
                      "
                    >

                      <p className="text-[10px] leading-5 text-slate-500">
                        <span className="text-amber-300">
                          Optical + SAR note:
                        </span>{" "}
                        These values are prototype indicators derived from RGB appearance and SAR intensity/texture. The fusion score is a heuristic indicator, not model accuracy or a calibrated probability. Image files without geospatial metadata cannot establish true co-registration on their own.
                      </p>

                    </div>

                  </div>

                </section>

              )}


              {/* =================================================
                  VISUAL EVIDENCE
              ================================================== */}

              {!analysisResult.change_analysis &&
                analysisResult.detected_task !== "grounding" &&
                analysisResult.detected_task !== "optical_sar_analysis" && (

                <div
                  className="
                    relative
                    mt-5
                    overflow-hidden
                    rounded-2xl
                    border
                    border-cyan-400/10
                    bg-[#040a12]
                    shadow-[0_20px_70px_rgba(0,0,0,0.3)]
                  "
                >

                  <div
                    className="
                      flex
                      items-center
                      justify-between
                      border-b
                      border-white/10
                      px-5
                      py-3
                    "
                  >

                    <div
                      className="
                        flex
                        items-center
                        gap-2
                      "
                    >

                      <ScanLine
                        className="
                          h-3.5
                          w-3.5
                          text-cyan-300
                        "
                      />

                      <span
                        className="
                          text-[10px]
                          uppercase
                          tracking-[0.16em]
                          text-slate-500
                        "
                      >
                        Visual Evidence Layer
                      </span>

                    </div>


                    <span
                      className="
                        text-[9px]
                        uppercase
                        tracking-widest
                        text-cyan-400
                      "
                    >
                      Analysis Feed
                    </span>

                  </div>


                  <div
                    className="
                      relative
                      flex
                      min-h-[350px]
                      items-center
                      justify-center
                      overflow-hidden
                      bg-[#02060c]
                    "
                  >

                    {image ? (

                      <img
                        src={image}
                        alt="Analysis result"
                        className="
                          max-h-[520px]
                          w-full
                          object-contain
                          opacity-90
                        "
                      />

                    ) : (

                      <div
                        className="
                          text-sm
                          text-slate-600
                        "
                      >
                        No imagery available
                      </div>

                    )}


                    {/* Scan line */}

                    <div
                      className="
                        pointer-events-none
                        absolute
                        inset-x-0
                        top-0
                        h-px
                        animate-[scanVertical_4s_linear_infinite]
                        bg-cyan-300/70
                        shadow-[0_0_20px_rgba(34,211,238,.8)]
                      "
                    />


                    {/* Grid */}

                    <div
                      className="
                        pointer-events-none
                        absolute
                        inset-0
                        opacity-[0.06]
                      "
                      style={{
                        backgroundImage:
                          "linear-gradient(rgba(34,211,238,.8) 1px, transparent 1px), linear-gradient(90deg, rgba(34,211,238,.8) 1px, transparent 1px)",
                        backgroundSize:
                          "80px 80px",
                      }}
                    />


                    <div
                      className="
                        pointer-events-none
                        absolute
                        inset-0
                        bg-gradient-to-t
                        from-[#02060c]/70
                        via-transparent
                        to-[#02060c]/10
                      "
                    />


                    <div
                      className="
                        absolute
                        left-4
                        top-4
                        rounded
                        border
                        border-cyan-400/20
                        bg-[#03101a]/80
                        px-2
                        py-1
                        font-mono
                        text-[9px]
                        text-cyan-300
                      "
                    >
                      VISUAL FEED // 01
                    </div>


                    <div
                      className="
                        absolute
                        right-4
                        top-4
                        font-mono
                        text-[9px]
                        text-slate-500
                      "
                    >
                      SATQUERY-VISION
                    </div>


                    <div
                      className="
                        absolute
                        bottom-4
                        left-4
                        rounded-lg
                        border
                        border-cyan-300/20
                        bg-[#06111d]/90
                        px-3
                        py-2
                        backdrop-blur
                      "
                    >

                      <p
                        className="
                          text-[9px]
                          uppercase
                          tracking-[0.18em]
                          text-cyan-300
                        "
                      >
                        Evidence Layer
                      </p>

                      <p
                        className="
                          mt-1
                          text-[10px]
                          text-slate-400
                        "
                      >
                        Input imagery received by
                        AI vision system
                      </p>

                    </div>

                  </div>

                </div>

              )}


              {/* =================================================
                  EVIDENCE TAGS
              ================================================== */}

              {analysisResult.evidence && (

                <div
                  className="
                    mt-4
                    flex
                    flex-wrap
                    gap-2
                  "
                >

                  {Object.entries(
                    analysisResult.evidence
                  ).map(
                    ([key, value]) => (

                      <EvidenceTag
                        key={key}
                        name={key}
                        value={value}
                      />

                    )
                  )}

                </div>

              )}


              {/* =================================================
                  ANSWER + TRACE
              ================================================== */}

              <div
                className="
                  mt-5
                  grid
                  gap-5
                  lg:grid-cols-3
                "
              >

                {/* AI RESPONSE */}

                <div
                  className="
                    relative
                    overflow-hidden
                    rounded-2xl
                    border
                    border-cyan-400/10
                    bg-white/[0.025]
                    p-6
                    lg:col-span-2
                  "
                >

                  <div
                    className="
                      absolute
                      right-0
                      top-0
                      h-40
                      w-40
                      rounded-full
                      bg-cyan-400/[0.035]
                      blur-3xl
                    "
                  />


                  <div className="relative">

                    <div
                      className="
                        flex
                        items-center
                        gap-2
                      "
                    >

                      <BrainCircuit
                        className="
                          h-4
                          w-4
                          text-cyan-300
                        "
                      />

                      <p
                        className="
                          text-[10px]
                          uppercase
                          tracking-[0.18em]
                          text-slate-500
                        "
                      >
                        SatQuery Intelligence
                      </p>

                    </div>


                    <p
                      className="
                        mt-5
                        text-base
                        leading-8
                        text-slate-200
                      "
                    >
                      {analysisResult.message}
                    </p>


                    <div
                      className="
                        mt-6
                        rounded-xl
                        border
                        border-white/10
                        bg-black/20
                        p-4
                      "
                    >

                      <div
                        className="
                          flex
                          items-center
                          gap-2
                        "
                      >

                        <span
                          className="
                            text-[9px]
                            uppercase
                            tracking-widest
                            text-slate-600
                          "
                        >
                          Query submitted
                        </span>

                      </div>


                      <p
                        className="
                          mt-2
                          text-xs
                          text-slate-400
                        "
                      >
                        "{analysisResult.query}"
                      </p>

                    </div>

                  </div>

                </div>


                {/* =================================================
                    EXECUTION TRACE
                ================================================== */}

                <div
                  className="
                    relative
                    overflow-hidden
                    rounded-2xl
                    border
                    border-cyan-400/10
                    bg-white/[0.025]
                    p-6
                  "
                >

                  <div
                    className="
                      flex
                      items-center
                      gap-2
                    "
                  >

                    <Activity
                      className="
                        h-4
                        w-4
                        text-cyan-300
                      "
                    />

                    <p
                      className="
                        text-[10px]
                        uppercase
                        tracking-[0.18em]
                        text-slate-500
                      "
                    >
                      Execution Trace
                    </p>

                  </div>


                  <div
                    className="
                      mt-6
                      space-y-5
                    "
                  >

                    {(
                      analysisResult.execution_trace ||
                      []
                    ).map(
                      (trace, index) => (

                        <div
                          key={`${trace.step}-${index}`}
                          className="
                            flex
                            items-start
                            gap-3
                          "
                        >

                          <div
                            className="relative"
                          >

                            {trace.status ===
                              "error" ? (

                              <span
                                className="
                                  flex
                                  h-4
                                  w-4
                                  items-center
                                  justify-center
                                  rounded-full
                                  border
                                  border-red-400
                                  text-[8px]
                                  text-red-400
                                "
                              >
                                !
                              </span>

                            ) : (

                              <CheckCircle2
                                className="
                                  h-4
                                  w-4
                                  text-emerald-400
                                "
                              />

                            )}


                            {index <
                              (
                                analysisResult
                                  .execution_trace
                                  ?.length || 1
                              ) - 1 && (

                              <div
                                className="
                                  absolute
                                  left-[7px]
                                  top-5
                                  h-5
                                  w-px
                                  bg-emerald-400/15
                                "
                              />

                            )}

                          </div>


                          <div>

                            <p
                              className="
                                text-xs
                                text-slate-300
                              "
                            >
                              {trace.step}
                            </p>


                            {trace.model && (

                              <p
                                className="
                                  mt-1
                                  text-[9px]
                                  text-slate-600
                                "
                              >
                                {trace.model}
                              </p>

                            )}


                            {trace.task && (

                              <p
                                className="
                                  mt-1
                                  text-[9px]
                                  uppercase
                                  tracking-wider
                                  text-cyan-400
                                "
                              >
                                Task:{" "}
                                {formatTaskName(
                                  trace.task
                                )}
                              </p>

                            )}

                          </div>

                        </div>

                      )
                    )}

                  </div>


                  <button
                    className="
                      mt-7
                      flex
                      items-center
                      gap-1
                      text-[10px]
                      uppercase
                      tracking-wider
                      text-cyan-300
                      transition
                      hover:text-cyan-200
                    "
                  >

                    View execution details

                    <ChevronRight
                      className="h-3 w-3"
                    />

                  </button>

                </div>

              </div>

            </section>

          )}

      </main>


      {/* =====================================================
          FOOTER
      ====================================================== */}

      <footer
        className="
          relative
          z-10
          border-t
          border-cyan-400/5
          py-7
          text-center
        "
      >

        <div
          className="
            flex
            items-center
            justify-center
            gap-3
          "
        >

          <Satellite
            className="
              h-3
              w-3
              text-cyan-400/50
            "
          />

          <p
            className="
              text-[9px]
              uppercase
              tracking-[0.28em]
              text-slate-700
            "
          >
            SatQuery AI · Orbital Intelligence Platform
          </p>

          <Satellite
            className="
              h-3
              w-3
              text-cyan-400/50
            "
          />

        </div>

      </footer>


      {/* =====================================================
          ANIMATION STYLES
      ====================================================== */}

      <style>{`

        @keyframes scan {

          0% {
            transform: translateX(-100%);
          }

          100% {
            transform: translateX(400%);
          }

        }


        @keyframes scanVertical {

          0% {
            transform: translateY(-10px);
          }

          100% {
            transform: translateY(520px);
          }

        }


        @keyframes fadeUp {

          from {
            opacity: 0;
            transform: translateY(18px);
          }

          to {
            opacity: 1;
            transform: translateY(0);
          }

        }

      `}</style>

    </div>
  );
}


// ============================================================
// STATUS PILL
// ============================================================

function StatusPill({
  icon,
  text,
}: {
  icon: React.ReactNode;
  text: string;
}) {

  return (

    <div
      className="
        flex
        items-center
        gap-2
        rounded-full
        border
        border-white/10
        bg-white/[0.02]
        px-3
        py-1.5
      "
    >

      <span className="text-cyan-400">
        {icon}
      </span>

      <span
        className="
          text-[9px]
          uppercase
          tracking-wider
          text-slate-600
        "
      >
        {text}
      </span>

    </div>

  );
}


// ============================================================
// IMAGE DROP ZONE
// ============================================================

function ImageDropZone({
  id,
  image,
  label,
  emptyTitle,
  emptySubtitle,
  icon,
  onChange,
  onRemove,
}: {
  id: string;

  image: string | null;

  label: string;

  emptyTitle: string;

  emptySubtitle: string;

  icon: React.ReactNode;

  onChange: (
    event: React.ChangeEvent<HTMLInputElement>
  ) => void;

  onRemove: () => void;
}) {

  return (

    <div className="relative">

      <input
        id={id}
        type="file"
        accept=".png,.jpg,.jpeg,.tif,.tiff,.webp"
        className="hidden"
        onChange={onChange}
      />


      <label
        htmlFor={id}
        className="
          group
          relative
          block
          aspect-video
          cursor-pointer
          overflow-hidden
          rounded-xl
          border
          border-dashed
          border-white/10
          bg-[#040a13]
          transition-all
          hover:border-cyan-400/40
          hover:bg-cyan-400/[0.015]
        "
      >

        {image ? (

          <>

            <img
              src={image}
              alt={label}
              className="
                h-full
                w-full
                object-cover
                transition
                duration-700
                group-hover:scale-[1.03]
              "
            />


            <div
              className="
                absolute
                inset-0
                bg-gradient-to-t
                from-black/80
                via-transparent
                to-black/10
              "
            />


            <div
              className="
                absolute
                left-0
                right-0
                top-0
                h-px
                animate-[scanVertical_5s_linear_infinite]
                bg-cyan-300/50
                shadow-[0_0_15px_rgba(34,211,238,.7)]
              "
            />


            <button
              onClick={(e) => {

                e.preventDefault();

                onRemove();
              }}
              className="
                absolute
                right-2
                top-2
                rounded-lg
                border
                border-white/10
                bg-black/70
                p-1.5
                text-white
                backdrop-blur
                transition
                hover:border-red-400/30
                hover:bg-red-950/60
              "
            >

              <X
                className="h-3.5 w-3.5"
              />

            </button>


            <div
              className="
                absolute
                bottom-0
                left-0
                right-0
                px-3
                py-2.5
              "
            >

              <div
                className="
                  flex
                  items-center
                  gap-2
                "
              >

                <span
                  className="
                    h-1.5
                    w-1.5
                    rounded-full
                    bg-emerald-400
                    shadow-[0_0_8px_rgba(52,211,153,.8)]
                  "
                />

                <span
                  className="
                    text-[9px]
                    uppercase
                    tracking-widest
                    text-emerald-300
                  "
                >
                  {label}
                </span>

              </div>

            </div>

          </>

        ) : (

          <div
            className="
              flex
              h-full
              flex-col
              items-center
              justify-center
            "
          >

            <div
              className="
                mb-3
                flex
                h-11
                w-11
                items-center
                justify-center
                rounded-xl
                border
                border-white/10
                bg-white/[0.025]
                text-slate-500
                transition-all
                group-hover:border-cyan-400/30
                group-hover:bg-cyan-400/[0.05]
                group-hover:text-cyan-300
              "
            >
              {icon}
            </div>


            <p
              className="
                text-xs
                font-medium
                text-slate-400
              "
            >
              {emptyTitle}
            </p>


            <p
              className="
                mt-1
                text-[9px]
                uppercase
                tracking-wider
                text-slate-700
              "
            >
              {emptySubtitle}
            </p>

          </div>

        )}

      </label>

    </div>

  );
}


// ============================================================
// RESULT CARD
// ============================================================

function ResultCard({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;

  label: string;

  value: string;
}) {

  return (

    <div
      className="
        group
        relative
        overflow-hidden
        rounded-xl
        border
        border-white/10
        bg-white/[0.025]
        p-4
        transition-all
        hover:border-cyan-400/20
        hover:bg-cyan-400/[0.015]
      "
    >

      <div
        className="
          absolute
          right-0
          top-0
          h-16
          w-16
          rounded-full
          bg-cyan-400/[0.03]
          blur-2xl
          transition
          group-hover:bg-cyan-400/[0.07]
        "
      />


      <div className="relative">

        <div
          className="
            flex
            items-center
            gap-2
            text-cyan-400/70
          "
        >

          {icon}

          <span
            className="
              text-[9px]
              uppercase
              tracking-[0.16em]
              text-slate-600
            "
          >
            {label}
          </span>

        </div>


        <p
          className="
            mt-3
            text-sm
            font-medium
            text-slate-200
          "
        >
          {value}
        </p>

      </div>

    </div>

  );
}


// ============================================================
// CHANGE STAT CARD
// ============================================================

function ChangeStatCard({
  label,
  value,
  description,
}: {
  label: string;
  value: string;
  description: string;
}) {

  return (

    <div
      className="
        rounded-xl
        border
        border-white/10
        bg-white/[0.025]
        p-4
      "
    >

      <p
        className="
          text-[9px]
          uppercase
          tracking-[0.16em]
          text-slate-600
        "
      >
        {label}
      </p>


      <p
        className="
          mt-2
          text-2xl
          font-semibold
          text-cyan-300
        "
      >
        {value}
      </p>


      <p
        className="
          mt-1
          text-[9px]
          uppercase
          tracking-wider
          text-slate-600
        "
      >
        {description}
      </p>

    </div>

  );
}


// ============================================================
// EVIDENCE VALUE BADGE
// ============================================================

function EvidenceValueBadge({
  value,
}: {
  value: string;
}) {

  const normalized = value.toLowerCase();

  const positive =
    normalized === "yes" ||
    normalized === "present" ||
    normalized === "dense";

  return (

    <span
      className={`
        inline-flex
        rounded-full
        border
        px-2.5
        py-1
        text-[9px]
        uppercase
        tracking-wider
        ${
          positive
            ? "border-emerald-400/20 bg-emerald-400/5 text-emerald-300"
            : "border-slate-400/10 bg-slate-400/5 text-slate-500"
        }
      `}
    >
      {value}
    </span>
  );
}


// ============================================================
// EVIDENCE TAG
// ============================================================

function EvidenceTag({
  name,
  value,
}: {
  name: string;

  value: string;
}) {

  const lower =
    value.toLowerCase();


  let icon =
    <Map className="h-3 w-3" />;


  if (
    name === "vegetation" ||
    name === "forest"
  ) {

    icon =
      <TreePine className="h-3 w-3" />;
  }


  if (name === "water") {

    icon =
      <Waves className="h-3 w-3" />;
  }


  if (
    name === "buildings" ||
    name === "urban"
  ) {

    icon =
      <Building2 className="h-3 w-3" />;
  }


  return (

    <div
      className="
        flex
        items-center
        gap-2
        rounded-full
        border
        border-white/10
        bg-white/[0.025]
        px-3
        py-1.5
      "
    >

      <span
        className="
          text-cyan-400/70
        "
      >
        {icon}
      </span>


      <span
        className="
          text-[9px]
          uppercase
          tracking-wider
          text-slate-600
        "
      >
        {name}
      </span>


      <span
        className="
          text-[9px]
          font-medium
          uppercase
          tracking-wider
          text-slate-300
        "
      >
        {lower}
      </span>

    </div>

  );
}


export default App;