#!/usr/bin/env bash
set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
DRY_RUN=false
GEMMA_ONLY=false
declare -a SELECTED_GROUPS=()
declare -a RAW_SELECTED_MODELS=()
declare -a RAW_SELECTED_SUITES=()
declare -a SELECTED_MODELS=()
declare -a SELECTED_SUITES=()

GROUP="${GROUP:-olmo-eval-olmo3-baselines-04272026}"
WORKSPACE="${WORKSPACE:-ai2/olmo-eval-debug}"
BUDGET="${BUDGET:-ai2/oe-other}"
CLUSTER="${CLUSTER:-h100}"
EXEC_HARNESS="${EXEC_HARNESS:-${HARNESS:-codex_universal}}"
NON_EXEC_HARNESS="${NON_EXEC_HARNESS:-default}"
TASK_PRIORITY="${TASK_PRIORITY:-urgent}"
MODAL_ENVIRONMENT="${MODAL_ENVIRONMENT:-oe-eval}"
PROVIDER_NUM_INSTANCES="${PROVIDER_NUM_INSTANCES:-8}"

usage() {
    cat <<EOF
Usage: ${SCRIPT_NAME} [--dry-run] [--gemma-only] [--only-group GROUP ...] [--only-suite SUITE ...] [--only-model MODEL ...]

Launches Beaker variants for the OLMo 3 baseline sweep across:
  1. Code execution tasks
  2. MCQA stem tasks
  3. MCQA non-stem tasks
  4. Generation tasks
  5. Math tasks
  6. Easy QA tasks
  7. Easy math + easy code tasks

Each task group is launched for the standard baseline model bundle, for Qwen3 Base,
for Nemotron Nano with prefix caching disabled, and for Gemma.

Options:
  --dry-run     Print the commands without launching them
  --gemma-only  Launch only the Gemma variants
  --only-group  Restrict launches to specific task groups (repeatable)
  --only-suite  Restrict launches to specific suites (repeatable)
  --only-model  Restrict launches to specific models (repeatable)
  --help        Show this help

Valid group names:
  code_exec, mcqa_stem, mcqa_non_stem, gen, math, easy_qa, easy_math_code

Legacy aliases:
  mcqa      -> mcqa_stem + mcqa_non_stem
  gen_math  -> gen + math

Valid suite names:
  olmobase:code
  olmobase:easy:code:bpb
  olmobase:easy:math:bpb
  olmobase:easy:qa:bpb
  olmobase:easy:qa:rc
  olmobase:gen
  olmobase:math
  olmobase:mcqa_non_stem
  olmobase:mcqa_stem

Models may be specified as full Hugging Face ids or trailing names such as
gemma-2-9b, qwen3-8b-base, or mimo-7b-base.

Examples:
  ${SCRIPT_NAME} --gemma-only --only-group mcqa_stem --only-group math
  ${SCRIPT_NAME} --only-group code_exec
  ${SCRIPT_NAME} --only-suite olmobase:math --only-suite olmobase:code
  ${SCRIPT_NAME} --only-model gemma-2-9b --only-model qwen3-8b-base --only-group gen

Environment overrides:
  GROUP=${GROUP}
  WORKSPACE=${WORKSPACE}
  BUDGET=${BUDGET}
  CLUSTER=${CLUSTER}
  EXEC_HARNESS=${EXEC_HARNESS}
  NON_EXEC_HARNESS=${NON_EXEC_HARNESS}
  TASK_PRIORITY=${TASK_PRIORITY}
  MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT}
  PROVIDER_NUM_INSTANCES=${PROVIDER_NUM_INSTANCES}
EOF
}

add_selected_groups_for_input() {
    case "$1" in
        code_exec|code-exec|code|exec)
            add_selected_group "code_exec"
            ;;
        mcqa)
            add_selected_group "mcqa_stem"
            add_selected_group "mcqa_non_stem"
            ;;
        mcqa_stem|mcqa-stem)
            add_selected_group "mcqa_stem"
            ;;
        mcqa_non_stem|mcqa-non-stem|mcqa_nonstem|mcqa-nonstem)
            add_selected_group "mcqa_non_stem"
            ;;
        gen_math|gen-math|gen+math)
            add_selected_group "gen"
            add_selected_group "math"
            ;;
        gen)
            add_selected_group "gen"
            ;;
        math)
            add_selected_group "math"
            ;;
        easy_qa|easy-qa)
            add_selected_group "easy_qa"
            ;;
        easy_math_code|easy-math-code|easy_math|easy-math|easy_code|easy-code)
            add_selected_group "easy_math_code"
            ;;
        *)
            return 1
            ;;
    esac
}

add_selected_group() {
    local group=$1
    local existing

    for existing in "${SELECTED_GROUPS[@]-}"; do
        if [[ "${existing}" == "${group}" ]]; then
            return 0
        fi
    done

    SELECTED_GROUPS+=("${group}")
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --gemma-only)
            GEMMA_ONLY=true
            shift
            ;;
        --only-group)
            if [[ $# -lt 2 ]]; then
                echo "Error: --only-group requires a value." >&2
                exit 1
            fi
            if ! add_selected_groups_for_input "$2"; then
                echo "Error: Unknown group '$2'." >&2
                echo "Valid groups: code_exec, mcqa_stem, mcqa_non_stem, gen, math, easy_qa, easy_math_code" >&2
                exit 1
            fi
            shift 2
            ;;
        --only-suite)
            if [[ $# -lt 2 ]]; then
                echo "Error: --only-suite requires a value." >&2
                exit 1
            fi
            RAW_SELECTED_SUITES+=("$2")
            shift 2
            ;;
        --only-model)
            if [[ $# -lt 2 ]]; then
                echo "Error: --only-model requires a value." >&2
                exit 1
            fi
            RAW_SELECTED_MODELS+=("$2")
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            echo "Use --help for usage." >&2
            exit 1
            ;;
    esac
done

group_is_selected() {
    local wanted=$1
    local selected

    if [[ "${#SELECTED_GROUPS[@]}" -eq 0 ]]; then
        return 0
    fi

    for selected in "${SELECTED_GROUPS[@]-}"; do
        if [[ "${selected}" == "${wanted}" ]]; then
            return 0
        fi
    done

    return 1
}

code_exec_tasks=(
    "olmobase:code"
)

mcqa_stem_tasks=(
    "olmobase:mcqa_stem"
)

mcqa_non_stem_tasks=(
    "olmobase:mcqa_non_stem"
)

gen_tasks=(
    "olmobase:gen"
)

math_tasks=(
    "olmobase:math"
)

non_exec_easy_qa_tasks=(
    "olmobase:easy:qa:rc"
    "olmobase:easy:qa:bpb"
)

non_exec_easy_math_code_tasks=(
    "olmobase:easy:math:bpb"
    "olmobase:easy:code:bpb"
)

baseline_models=(
    "allenai/OLMo-2-1124-7B"
    "allenai/Olmo-3-1025-7B"
    "marin-community/marin-8b-base"
    "swiss-ai/Apertus-8B-2509"
    "almanach/Gaperon-1125-8B"
    "Qwen/Qwen3-8B-Base"
    "Qwen/Qwen2.5-7B"
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
    "ibm-granite/granite-3.3-8b-base"
    "XiaomiMiMo/MiMo-7B-Base"
)

gemma_models=(
    "google/gemma-2-9b"
)

NEMOTRON_NANO_MODEL="nvidia/NVIDIA-Nemotron-Nano-9B-v2"
QWEN3_MODEL="Qwen/Qwen3-8B-Base"

baseline_launch_args=(
    "-o" "provider.num_instances=${PROVIDER_NUM_INSTANCES}"
    "-o" "provider.trust_remote_code=true"
)

qwen3_launch_args=(
    "-o" "provider.num_instances=${PROVIDER_NUM_INSTANCES}"
    "-o" "provider.trust_remote_code=true"
)

nemotron_launch_args=(
    "-o" "provider.num_instances=${PROVIDER_NUM_INSTANCES}"
    "-o" "provider.trust_remote_code=true"
    "-o" "provider.kwargs.enable_prefix_caching=false"
)

gemma_launch_args=(
    "-o" "provider.num_instances=${PROVIDER_NUM_INSTANCES}"
    "-o" "provider.kwargs.attention_backend=TRITON_ATTN"
)

exec_only_args=(
    "-o" 'sandboxes={"mode":"modal","instances":64, "min_instances": 56, "registry_auth":{"provider":"gcp"}}'
    "-e" "MODAL_ENVIRONMENT=${MODAL_ENVIRONMENT}"
    "--secret-env" "olmo-eval_MODAL_TOKEN_ID:MODAL_TOKEN_ID"
    "--secret-env" "olmo-eval_MODAL_TOKEN_SECRET:MODAL_TOKEN_SECRET"
)

common_tail_args=(
    "-w" "${WORKSPACE}"
    "-B" "${BUDGET}"
    "--cluster" "${CLUSTER}"
    "--group" "${GROUP}"
    "--upload"
    "--inspect"
    "--gcp-credentials"
    "--no-follow"
    "-y"
)

to_lower() {
    printf "%s" "$1" | tr '[:upper:]' '[:lower:]'
}

normalize_suite_name() {
    case "$1" in
        olmobase:code|code)
            printf "olmobase:code"
            ;;
        olmobase:mcqa_stem|mcqa_stem|mcqa-stem)
            printf "olmobase:mcqa_stem"
            ;;
        olmobase:mcqa_non_stem|mcqa_non_stem|mcqa-non-stem|mcqa_nonstem|mcqa-nonstem)
            printf "olmobase:mcqa_non_stem"
            ;;
        olmobase:gen|gen)
            printf "olmobase:gen"
            ;;
        olmobase:math|math)
            printf "olmobase:math"
            ;;
        olmobase:easy:qa:rc|easy:qa:rc|easy_qa_rc|easy-qa-rc)
            printf "olmobase:easy:qa:rc"
            ;;
        olmobase:easy:qa:bpb|easy:qa:bpb|easy_qa_bpb|easy-qa-bpb)
            printf "olmobase:easy:qa:bpb"
            ;;
        olmobase:easy:math:bpb|easy:math:bpb|easy_math_bpb|easy-math-bpb)
            printf "olmobase:easy:math:bpb"
            ;;
        olmobase:easy:code:bpb|easy:code:bpb|easy_code_bpb|easy-code-bpb)
            printf "olmobase:easy:code:bpb"
            ;;
        *)
            return 1
            ;;
    esac
}

normalize_model_name() {
    local candidate=$1
    local candidate_lower
    local model
    local model_lower
    local short_lower

    candidate_lower="$(to_lower "${candidate}")"

    case "${candidate_lower}" in
        gemma|gemma-2|gemma2|gemma-2-9b)
            printf "google/gemma-2-9b"
            return 0
            ;;
        qwen3|qwen3-8b|qwen3-base|qwen3-8b-base)
            printf "Qwen/Qwen3-8B-Base"
            return 0
            ;;
    esac

    for model in "${baseline_models[@]}" "${gemma_models[@]}"; do
        model_lower="$(to_lower "${model}")"
        short_lower="$(to_lower "${model##*/}")"
        if [[ "${candidate_lower}" == "${model_lower}" || "${candidate_lower}" == "${short_lower}" ]]; then
            printf "%s" "${model}"
            return 0
        fi
    done

    return 1
}

print_valid_suites() {
    local suite

    for suite in \
        "${code_exec_tasks[@]}" \
        "${mcqa_stem_tasks[@]}" \
        "${mcqa_non_stem_tasks[@]}" \
        "${gen_tasks[@]}" \
        "${math_tasks[@]}" \
        "${non_exec_easy_qa_tasks[@]}" \
        "${non_exec_easy_math_code_tasks[@]}"; do
        echo "  ${suite}" >&2
    done
}

print_valid_models() {
    local model

    for model in "${baseline_models[@]}" "${gemma_models[@]}"; do
        echo "  ${model}" >&2
    done
}

add_selected_suite() {
    local suite=$1
    local existing

    for existing in "${SELECTED_SUITES[@]-}"; do
        if [[ "${existing}" == "${suite}" ]]; then
            return 0
        fi
    done

    SELECTED_SUITES+=("${suite}")
}

add_selected_model() {
    local model=$1
    local existing

    for existing in "${SELECTED_MODELS[@]-}"; do
        if [[ "${existing}" == "${model}" ]]; then
            return 0
        fi
    done

    SELECTED_MODELS+=("${model}")
}

suite_is_selected() {
    local wanted=$1
    local selected

    if [[ "${#SELECTED_SUITES[@]}" -eq 0 ]]; then
        return 0
    fi

    for selected in "${SELECTED_SUITES[@]-}"; do
        if [[ "${selected}" == "${wanted}" ]]; then
            return 0
        fi
    done

    return 1
}

model_is_selected() {
    local wanted=$1
    local selected

    if [[ "${#SELECTED_MODELS[@]}" -eq 0 ]]; then
        return 0
    fi

    for selected in "${SELECTED_MODELS[@]-}"; do
        if [[ "${selected}" == "${wanted}" ]]; then
            return 0
        fi
    done

    return 1
}

filter_tasks() {
    local target=$1
    shift
    local filtered=()
    local task

    for task in "$@"; do
        if suite_is_selected "${task}"; then
            filtered+=("${task}")
        fi
    done

    if [[ "${#filtered[@]}" -gt 0 ]]; then
        eval "${target}=(\"\${filtered[@]}\")"
    else
        eval "${target}=()"
    fi
}

filter_models() {
    local target=$1
    shift
    local filtered=()
    local model

    for model in "$@"; do
        if model_is_selected "${model}"; then
            filtered+=("${model}")
        fi
    done

    if [[ "${#filtered[@]}" -gt 0 ]]; then
        eval "${target}=(\"\${filtered[@]}\")"
    else
        eval "${target}=()"
    fi
}

if [[ "${#RAW_SELECTED_SUITES[@]}" -gt 0 ]]; then
    for raw_selected_suite in "${RAW_SELECTED_SUITES[@]}"; do
        if ! normalized_suite="$(normalize_suite_name "${raw_selected_suite}")"; then
            echo "Error: Unknown suite '${raw_selected_suite}'." >&2
            echo "Valid suites:" >&2
            print_valid_suites
            exit 1
        fi
        add_selected_suite "${normalized_suite}"
    done
fi

if [[ "${#RAW_SELECTED_MODELS[@]}" -gt 0 ]]; then
    for raw_selected_model in "${RAW_SELECTED_MODELS[@]}"; do
        if ! normalized_model="$(normalize_model_name "${raw_selected_model}")"; then
            echo "Error: Unknown model '${raw_selected_model}'." >&2
            echo "Valid models:" >&2
            print_valid_models
            exit 1
        fi
        add_selected_model "${normalized_model}"
    done
fi

declare -a selected_code_exec_tasks=()
declare -a selected_mcqa_stem_tasks=()
declare -a selected_mcqa_non_stem_tasks=()
declare -a selected_gen_tasks=()
declare -a selected_math_tasks=()
declare -a selected_non_exec_easy_qa_tasks=()
declare -a selected_non_exec_easy_math_code_tasks=()
declare -a selected_baseline_models=()
declare -a selected_standard_baseline_models=()
declare -a selected_qwen3_models=()
declare -a selected_nemotron_nano_models=()
declare -a selected_gemma_models=()

filter_tasks selected_code_exec_tasks "${code_exec_tasks[@]}"
filter_tasks selected_mcqa_stem_tasks "${mcqa_stem_tasks[@]}"
filter_tasks selected_mcqa_non_stem_tasks "${mcqa_non_stem_tasks[@]}"
filter_tasks selected_gen_tasks "${gen_tasks[@]}"
filter_tasks selected_math_tasks "${math_tasks[@]}"
filter_tasks selected_non_exec_easy_qa_tasks "${non_exec_easy_qa_tasks[@]}"
filter_tasks selected_non_exec_easy_math_code_tasks "${non_exec_easy_math_code_tasks[@]}"
filter_models selected_baseline_models "${baseline_models[@]}"
filter_models selected_gemma_models "${gemma_models[@]}"

for model in ${selected_baseline_models[@]+"${selected_baseline_models[@]}"}; do
    if [[ "${model}" == "${NEMOTRON_NANO_MODEL}" ]]; then
        selected_nemotron_nano_models+=("${model}")
    elif [[ "${model}" == "${QWEN3_MODEL}" ]]; then
        selected_qwen3_models+=("${model}")
    else
        selected_standard_baseline_models+=("${model}")
    fi
done

declare -a current_cmd

start_command() {
    local harness=$1

    current_cmd=(
        "olmo-eval" "beaker" "launch"
        "-H" "${harness}"
    )
}

append_args() {
    current_cmd+=("$@")
}

append_models() {
    local model

    for model in "$@"; do
        current_cmd+=("-m" "${model}")
    done
}

append_tasks() {
    local task

    for task in "$@"; do
        current_cmd+=("-t" "${task}@${TASK_PRIORITY}")
    done
}

build_variant_cmd() {
    local harness=$1
    local launch_args_name=$2
    local models_name=$3
    local tasks_name=$4
    local include_exec_only=${5:-false}

    start_command "${harness}"
    eval "append_args \"\${${launch_args_name}[@]}\""
    if [[ "${include_exec_only}" == "true" ]]; then
        append_args "${exec_only_args[@]}"
    fi
    eval "append_models \${${models_name}[@]+\"\${${models_name}[@]}\"}"
    eval "append_tasks \${${tasks_name}[@]+\"\${${tasks_name}[@]}\"}"
    append_args "${common_tail_args[@]}"
}

print_command() {
    local cmd=("$@")
    local lines=()
    local index=3
    local last_line_index

    lines+=("$(format_tokens "${cmd[@]:0:3}")")

    while [[ "${index}" -lt "${#cmd[@]}" ]]; do
        if arg_takes_value "${cmd[${index}]}"; then
            lines+=("$(format_tokens "${cmd[${index}]}" "${cmd[$((index + 1))]}")")
            index=$((index + 2))
        else
            lines+=("$(format_tokens "${cmd[${index}]}")")
            index=$((index + 1))
        fi
    done

    last_line_index=$((${#lines[@]} - 1))

    for index in "${!lines[@]}"; do
        if [[ "${index}" -eq 0 ]]; then
            if [[ "${index}" -lt "${last_line_index}" ]]; then
                printf '%s \
' "${lines[${index}]}"
            else
                printf "%s\n" "${lines[${index}]}"
            fi
        else
            if [[ "${index}" -lt "${last_line_index}" ]]; then
                printf '  %s \
' "${lines[${index}]}"
            else
                printf "  %s\n" "${lines[${index}]}"
            fi
        fi
    done
}

arg_takes_value() {
    case "$1" in
        -H|-o|-e|--secret-env|-m|-t|-w|-B|--cluster|--group)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

quote_token() {
    local token=$1
    local escaped

    if [[ "${token}" =~ ^[A-Za-z0-9_./:@=%,+-]+$ ]]; then
        printf "%s" "${token}"
    else
        escaped=${token//\'/\'\\\'\'}
        printf "'%s'" "${escaped}"
    fi
}

format_tokens() {
    local token
    local formatted=""
    local quoted

    for token in "$@"; do
        quoted="$(quote_token "${token}")"
        if [[ -n "${formatted}" ]]; then
            formatted="${formatted} ${quoted}"
        else
            formatted="${quoted}"
        fi
    done

    printf "%s" "${formatted}"
}

run_variant() {
    local label=$1
    shift
    local cmd=("$@")

    echo "========================================="
    echo "${label}"
    echo "========================================="
    print_command "${cmd[@]}"
    echo ""

    if [[ "${DRY_RUN}" == "false" ]]; then
        "${cmd[@]}"
        echo ""
    fi
}

group_specs=(
    "code_exec|selected_code_exec_tasks|exec|true|code execution suites"
    "mcqa_stem|selected_mcqa_stem_tasks|non_exec|false|MCQA stem suites"
    "mcqa_non_stem|selected_mcqa_non_stem_tasks|non_exec|false|MCQA non-stem suites"
    "gen|selected_gen_tasks|non_exec|false|generation suites"
    "math|selected_math_tasks|non_exec|false|math suites"
    "easy_qa|selected_non_exec_easy_qa_tasks|non_exec|false|easy QA suites"
    "easy_math_code|selected_non_exec_easy_math_code_tasks|non_exec|false|easy math + easy code suites"
)

family_specs=(
    "baseline|selected_standard_baseline_models|baseline_launch_args|Baseline models|false|"
    "qwen3|selected_qwen3_models|qwen3_launch_args|Qwen3 Base|false|"
    "nemotron|selected_nemotron_nano_models|nemotron_launch_args|Nemotron Nano|false| (prefix caching disabled)"
    "gemma|selected_gemma_models|gemma_launch_args|Gemma|true|"
)

RESOLVED_TASKS_NAME=""
RESOLVED_HARNESS_KIND=""
RESOLVED_INCLUDE_EXEC_ONLY=""
RESOLVED_GROUP_LABEL=""
RESOLVED_MODELS_NAME=""
RESOLVED_LAUNCH_ARGS_NAME=""
RESOLVED_LABEL_PREFIX=""
RESOLVED_ALLOW_WHEN_GEMMA_ONLY=""
RESOLVED_FAMILY_LABEL_SUFFIX=""

resolve_group_spec() {
    local wanted=$1
    local spec
    local group_name
    local tasks_name
    local harness_kind
    local include_exec_only
    local group_label

    for spec in "${group_specs[@]}"; do
        IFS='|' read -r group_name tasks_name harness_kind include_exec_only group_label <<<"${spec}"
        if [[ "${group_name}" == "${wanted}" ]]; then
            RESOLVED_TASKS_NAME="${tasks_name}"
            RESOLVED_HARNESS_KIND="${harness_kind}"
            RESOLVED_INCLUDE_EXEC_ONLY="${include_exec_only}"
            RESOLVED_GROUP_LABEL="${group_label}"
            return 0
        fi
    done

    return 1
}

resolve_family_spec() {
    local wanted=$1
    local spec
    local family_name
    local models_name
    local launch_args_name
    local label_prefix
    local allow_when_gemma_only
    local family_label_suffix

    for spec in "${family_specs[@]}"; do
        IFS='|' read -r family_name models_name launch_args_name label_prefix allow_when_gemma_only family_label_suffix <<<"${spec}"
        if [[ "${family_name}" == "${wanted}" ]]; then
            RESOLVED_MODELS_NAME="${models_name}"
            RESOLVED_LAUNCH_ARGS_NAME="${launch_args_name}"
            RESOLVED_LABEL_PREFIX="${label_prefix}"
            RESOLVED_ALLOW_WHEN_GEMMA_ONLY="${allow_when_gemma_only}"
            RESOLVED_FAMILY_LABEL_SUFFIX="${family_label_suffix}"
            return 0
        fi
    done

    return 1
}

variant_is_enabled() {
    local family_name=$1
    local group_name=$2
    local model_count
    local task_count

    resolve_family_spec "${family_name}" || return 1
    resolve_group_spec "${group_name}" || return 1

    if [[ "${GEMMA_ONLY}" == "true" && "${RESOLVED_ALLOW_WHEN_GEMMA_ONLY}" != "true" ]]; then
        return 1
    fi

    if ! group_is_selected "${group_name}"; then
        return 1
    fi

    eval "model_count=\${#${RESOLVED_MODELS_NAME}[@]}"
    eval "task_count=\${#${RESOLVED_TASKS_NAME}[@]}"

    [[ "${model_count}" -gt 0 && "${task_count}" -gt 0 ]]
}

declare -a matched_variant_specs=()

queue_variants_for_families() {
    local group_spec
    local family_name
    local group_name

    for group_spec in "${group_specs[@]}"; do
        IFS='|' read -r group_name _ <<<"${group_spec}"
        for family_name in "$@"; do
            if variant_is_enabled "${family_name}" "${group_name}"; then
                matched_variant_specs+=("${family_name}|${group_name}")
                planned_variants=$((planned_variants + 1))
            fi
        done
    done
}

planned_variants=0

queue_variants_for_families baseline qwen3
queue_variants_for_families nemotron
queue_variants_for_families gemma

if [[ "${planned_variants}" -eq 0 ]]; then
    echo "Error: No launch variants matched the selected groups, suites, and models." >&2
    exit 1
fi

if [[ "${DRY_RUN}" == "true" ]]; then
    echo "Dry run enabled. Commands will be printed but not launched."
    echo ""
else
    if [[ "${GEMMA_ONLY}" == "true" ]]; then
        echo "Launching Gemma-only Beaker baseline variants..."
    else
        echo "Launching Beaker baseline variants..."
    fi
    echo ""
fi

if [[ "${#SELECTED_GROUPS[@]}" -gt 0 ]]; then
    echo "Selected groups: ${SELECTED_GROUPS[*]}"
    echo ""
fi

if [[ "${#SELECTED_SUITES[@]}" -gt 0 ]]; then
    echo "Selected suites: ${SELECTED_SUITES[*]}"
    echo ""
fi

if [[ "${#SELECTED_MODELS[@]}" -gt 0 ]]; then
    echo "Selected models: ${SELECTED_MODELS[*]}"
    echo ""
fi

for matched_variant_spec in "${matched_variant_specs[@]}"; do
    IFS='|' read -r matched_family_name matched_group_name <<<"${matched_variant_spec}"
    resolve_family_spec "${matched_family_name}"
    resolve_group_spec "${matched_group_name}"

    if [[ "${RESOLVED_HARNESS_KIND}" == "exec" ]]; then
        matched_harness="${EXEC_HARNESS}"
    else
        matched_harness="${NON_EXEC_HARNESS}"
    fi

    build_variant_cmd \
        "${matched_harness}" \
        "${RESOLVED_LAUNCH_ARGS_NAME}" \
        "${RESOLVED_MODELS_NAME}" \
        "${RESOLVED_TASKS_NAME}" \
        "${RESOLVED_INCLUDE_EXEC_ONLY}"
    run_variant \
        "${RESOLVED_LABEL_PREFIX}: ${RESOLVED_GROUP_LABEL}${RESOLVED_FAMILY_LABEL_SUFFIX}" \
        "${current_cmd[@]}"
done
