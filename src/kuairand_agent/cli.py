"""Public command-line interface for trusted local campaign lifecycle operations."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from kuairand_agent import __version__
from kuairand_agent.campaign import CampaignControllerError, CampaignEngine, CampaignStatus
from kuairand_agent.campaign.provenance import ProvenanceError, build_campaign_request
from kuairand_agent.config import (
    ConfigError,
    OpenAIFailoverResearchConfig,
    OpenAIResearchConfig,
    load_config,
)
from kuairand_agent.contract import OrganizerIntegrityError, verify_starter_kit
from kuairand_agent.data.acquire import (
    ArchiveIntegrityError,
    download_and_prepare,
    prepare_archive,
)
from kuairand_agent.data.audit import DataAuditError, audit_dataset, write_audit_report
from kuairand_agent.data.canonical import CanonicalDataError, load_canonical_dataset
from kuairand_agent.execution.signals import cancellation_on_signals
from kuairand_agent.finalization.iteration_log import (
    IterationLogError,
    build_iteration_log,
    render_jsonl,
    render_markdown,
)
from kuairand_agent.scoring.submission import AlignmentRow, SubmissionError, read_submission

EXIT_INVALID: Final = 2
EXIT_CONTRACT: Final = 3
DEFAULT_QUALIFICATION_RUN_DIR: Final = Path("runs/wp3-official-qualification")
DEFAULT_CAMPAIGN_ROOT: Final = Path("runs")


def _stable_json(value: object) -> str:
    """Render one finite, deterministic JSON value for automation-facing stdout."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _result_manifest(value: object, *, location: str) -> Mapping[str, object]:
    """Require one trusted manifest-bearing result before writing automation stdout."""

    manifest = getattr(value, "manifest", None)
    if not callable(manifest):
        raise RuntimeError(f"{location} did not return a manifest-bearing result")
    rendered = manifest()
    if not isinstance(rendered, Mapping) or any(type(key) is not str for key in rendered):
        raise RuntimeError(f"{location} returned an invalid result manifest")
    return rendered


def _sha256_argument(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("must be one lowercase SHA-256 digest")
    return value


def _validate_config(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.file)
    except ConfigError as exc:
        print(f"CONFIG_INVALID: {exc}", file=sys.stderr)
        return EXIT_INVALID
    payload = {"digest": config.digest, "effective_config": config.normalized()}
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


def _preflight_provider(args: argparse.Namespace) -> int:
    """Validate provider construction and credential availability without dispatching a call."""

    from kuairand_agent.research.factory import (
        ProviderUnavailableDiagnostic,
        select_research_provider,
    )

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"PROVIDER_PREFLIGHT_FAILED: {exc}", file=sys.stderr)
        return EXIT_INVALID
    selection = select_research_provider(config.research)
    if isinstance(selection, ProviderUnavailableDiagnostic):
        print(
            "PROVIDER_PREFLIGHT_FAILED: " + _stable_json(selection.to_wire()),
            file=sys.stderr,
        )
        return EXIT_INVALID
    openai = config.research.openai
    if isinstance(openai, OpenAIResearchConfig):
        model: str | None = openai.model
        credential_env: str | None = openai.api_key_env
        provider_profiles: list[dict[str, object]] = [
            {
                "slot": "main",
                "model": openai.model,
                "base_url": openai.base_url,
                "credential_env": openai.api_key_env,
            }
        ]
    elif isinstance(openai, OpenAIFailoverResearchConfig):
        model = None
        credential_env = None
        selected_model = selection.model
        endpoint_models = getattr(selected_model, "provider_models", ())
        provider_profiles = [
            {
                "slot": slot,
                "model": endpoint.config.model,
                "base_url": endpoint.config.base_url,
                "credential_env": endpoint.config.api_key_env,
            }
            for slot, endpoint in endpoint_models
        ]
    else:
        model = None
        credential_env = None
        provider_profiles = []
    payload = {
        "schema_version": 1,
        "status": "available",
        "config_digest": config.digest,
        "run_kind": config.research.run_kind,
        "provider": selection.provider,
        "live_provider_used": selection.live_provider_used,
        "model": model,
        "credential_env": credential_env,
        "provider_profiles": provider_profiles,
        "api_request_sent": False,
    }
    print(_stable_json(payload))
    return 0


def _verify_starter(args: argparse.Namespace) -> int:
    try:
        result = verify_starter_kit(args.starter_dir)
    except OrganizerIntegrityError as exc:
        print(f"CONTRACT_MISMATCH: {exc}", file=sys.stderr)
        return EXIT_CONTRACT
    print(
        json.dumps(
            {
                "root": str(result.root),
                "manifest_sha256": result.manifest_sha256,
                "files": result.files,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def _prepare_data(args: argparse.Namespace) -> int:
    """Verify and install the pinned archive into one new directory."""

    try:
        if args.download:
            result = download_and_prepare(args.data_dir)
        else:
            if args.archive is None:  # defended even though argparse makes this impossible
                raise ArchiveIntegrityError("--archive is required unless --download is used")
            result = prepare_archive(args.archive, args.data_dir)
    except ArchiveIntegrityError as exc:
        print(f"DATA_PREPARATION_FAILED: {exc}", file=sys.stderr)
        return EXIT_CONTRACT
    print(
        json.dumps(
            {
                "schema_version": 1,
                "archive_sha256": result.verification.archive_sha256,
                "destination": str(result.destination),
                "dataset_root": str(result.dataset_root),
                "data_dir": str(result.dataset_root / "data"),
                "integrity_manifest": str(result.integrity_manifest),
                "integrity_manifest_sha256": result.manifest_sha256,
                "member_count": len(result.verification.members),
                "payload_size": result.verification.payload_size,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def _audit_data(args: argparse.Namespace) -> int:
    """Run the streaming official-data audit without final-outcome access."""

    try:
        report = audit_dataset(args.data_dir)
    except DataAuditError as exc:
        print(f"DATA_AUDIT_FAILED: {exc}", file=sys.stderr)
        return EXIT_CONTRACT
    output_dir: Path | None = args.output_dir
    if output_dir is not None:
        if output_dir.exists():
            print(
                f"DATA_AUDIT_FAILED: output directory already exists: {output_dir}", file=sys.stderr
            )
            return EXIT_INVALID
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
            write_audit_report(
                report,
                json_path=output_dir / "data-audit.json",
                markdown_path=output_dir / "data-audit.md",
            )
        except OSError as exc:
            print(f"DATA_AUDIT_FAILED: cannot write evidence: {exc}", file=sys.stderr)
            return EXIT_INVALID
    if args.json:
        print(report.to_json())
    else:
        print(report.readable_report(), end="")
    return 0


def _qualify(args: argparse.Namespace) -> int:
    """Run the atomic six-launch official-baseline qualification."""

    from kuairand_agent.baselines.qualification import (
        QualificationError,
        QualificationRequest,
        run_qualification,
    )

    try:
        result = run_qualification(
            QualificationRequest(
                data_dir=args.data_dir,
                starter_dir=args.starter_dir,
                run_dir=args.run_dir,
            )
        )
    except QualificationError as exc:
        print(f"QUALIFICATION_FAILED: {exc}", file=sys.stderr)
        return EXIT_CONTRACT
    payload = {
        "schema_version": 1,
        "status": "baseline_reproduced",
        "run_dir": str(result.run_dir),
        "manifest_digest": result.manifest_digest,
        "fallback_seed": result.fallback_seed,
        "launch_count": result.launch_count,
        "validation_metrics": result.validation_metrics.manifest(),
        "validation_submission": str(result.validation_submission.path),
        "validation_submission_sha256": result.validation_submission.submission_digest,
        "final_submission": str(result.final_submission.path),
        "final_submission_sha256": result.final_submission.submission_digest,
        "final_outcomes_accessed": False,
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


def _validate_submission(args: argparse.Namespace) -> int:
    """Validate exact canonical alignment and finite high-precision scores."""

    if args.data_dir is None:
        print("SUBMISSION_INVALID: --data-dir is required for canonical alignment", file=sys.stderr)
        return EXIT_INVALID
    try:
        dataset = load_canonical_dataset(args.data_dir)
        split = dataset.valid if args.split == "valid" else dataset.final
        if args.split == "test" and split.targets is not None:
            raise SubmissionError("final split unexpectedly exposes a target capability")
        alignment = tuple(
            AlignmentRow(row_id, user_id, video_id)
            for row_id, user_id, video_id in zip(
                split.alignment.row_id,
                split.alignment.user_id,
                split.alignment.video_id,
                strict=True,
            )
        )
        checked = read_submission(args.file, alignment)
    except (CanonicalDataError, SubmissionError, OSError) as exc:
        print(f"SUBMISSION_INVALID: {exc}", file=sys.stderr)
        return EXIT_CONTRACT
    print(
        json.dumps(
            {
                "schema_version": 1,
                "split": args.split,
                "path": str(checked.path),
                "row_count": checked.row_count,
                "prediction_digest": checked.prediction_digest,
                "submission_sha256": checked.submission_digest,
                "canonical_alignment": True,
                "finite_scores": True,
                "final_outcomes_accessed": False,
                "organizer_check": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def _status_json(status: CampaignStatus) -> None:
    print(_stable_json(status.manifest()))


def _absolute_cli_path(path: Path) -> Path:
    """Make a CLI path absolute without resolving away a controller-visible symlink."""

    return path if path.is_absolute() else Path.cwd() / path


def _drive_provider_free_campaign(
    run_dir: Path,
    *,
    project_root: Path,
    engine: CampaignEngine,
    cancel_event: threading.Event,
) -> object:
    """Enter the fixed trusted production facade without widening its capability surface."""

    from kuairand_agent.campaign.full_campaign import run_provider_free_campaign

    return run_provider_free_campaign(
        run_dir,
        project_root=project_root,
        engine=engine,
        cancel_event=cancel_event,
    )


def _finalize_provider_free_campaign(
    run_dir: Path,
    *,
    project_root: Path,
    engine: CampaignEngine,
    cancel_event: threading.Event,
) -> object:
    """Delegate all campaign-to-finalization reconstruction to the trusted facade."""

    from kuairand_agent.finalization.production import finalize_provider_free_campaign

    return finalize_provider_free_campaign(
        run_dir,
        project_root=project_root,
        engine=engine,
        cancel_event=cancel_event,
    )


def _replay_final_bundle(
    bundle: Path,
    *,
    project_root: Path,
    data_dir: Path,
    expected_data_sha256: str,
    cancel_event: threading.Event,
) -> object:
    """Replay only a closed, verified bundle through the trusted provider-free facade."""

    from kuairand_agent.finalization.production import replay_final_bundle

    return replay_final_bundle(
        bundle,
        project_root=project_root,
        data_dir=data_dir,
        expected_data_sha256=expected_data_sha256,
        cancel_event=cancel_event,
    )


def _status(args: argparse.Namespace) -> int:
    """Read a campaign without reconciliation or any durable mutation."""

    try:
        status = CampaignEngine().status(_absolute_cli_path(args.run_dir))
    except (CampaignControllerError, OSError, RuntimeError) as exc:
        print(f"CAMPAIGN_STATUS_FAILED: {exc}", file=sys.stderr)
        return EXIT_CONTRACT
    if args.json:
        _status_json(status)
    else:
        print(
            f"{status.campaign_id}: {status.status} ({status.phase}); "
            f"launches {status.launches_used}/{status.launches_used + status.launches_remaining}; "
            f"incumbent {status.incumbent_id}"
        )
    return 0


def _iteration_log(args: argparse.Namespace) -> int:
    """Emit the required per-iteration run log from a campaign's own durable records."""

    try:
        entries = build_iteration_log(
            _absolute_cli_path(args.run_dir),
            project_root=Path.cwd().resolve(strict=True),
        )
    except (IterationLogError, OSError) as exc:
        print(f"ITERATION_LOG_FAILED: {exc}", file=sys.stderr)
        return EXIT_CONTRACT
    rendered = render_jsonl(entries) if args.format == "jsonl" else render_markdown(entries)
    if args.output is None:
        print(rendered, end="" if args.format == "jsonl" else "\n")
        return 0
    destination = _absolute_cli_path(args.output)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    except OSError as exc:
        print(f"ITERATION_LOG_FAILED: {exc}", file=sys.stderr)
        return EXIT_CONTRACT
    print(f"{destination}: {len(entries)} iteration(s)")
    return 0


def _resume(args: argparse.Namespace) -> int:
    """Reconcile abandoned executions and continue the original durable campaign."""

    project_root = Path.cwd().resolve(strict=True)
    run_dir = _absolute_cli_path(args.run_dir)
    engine = CampaignEngine()
    try:
        with cancellation_on_signals() as cancel_event:
            _drive_provider_free_campaign(
                run_dir,
                project_root=project_root,
                engine=engine,
                cancel_event=cancel_event,
            )
            finalized = _finalize_provider_free_campaign(
                run_dir,
                project_root=project_root,
                engine=engine,
                cancel_event=cancel_event,
            )
    except (CampaignControllerError, OSError, RuntimeError) as exc:
        print(f"CAMPAIGN_RESUME_FAILED: {exc}", file=sys.stderr)
        return EXIT_CONTRACT
    print(_stable_json(_result_manifest(finalized, location="campaign finalization")))
    return 0


def _finalize(args: argparse.Namespace) -> int:
    """Finalize the strictly retained outcome, walking back to the official FM if needed."""

    project_root = Path.cwd().resolve(strict=True)
    run_dir = _absolute_cli_path(args.run_dir)
    try:
        with cancellation_on_signals() as cancel_event:
            finalized = _finalize_provider_free_campaign(
                run_dir,
                project_root=project_root,
                engine=CampaignEngine(),
                cancel_event=cancel_event,
            )
        payload = _result_manifest(finalized, location="campaign finalization")
    except (CampaignControllerError, OSError, RuntimeError) as exc:
        print(f"CAMPAIGN_FINALIZE_FAILED: {exc}", file=sys.stderr)
        return EXIT_CONTRACT
    print(_stable_json(payload))
    return 0


def _replay(args: argparse.Namespace) -> int:
    """Replay a closed SHA-bound bundle with canonical label-free capabilities."""

    try:
        with cancellation_on_signals() as cancel_event:
            result = _replay_final_bundle(
                _absolute_cli_path(args.bundle),
                project_root=_absolute_cli_path(args.project_root),
                data_dir=_absolute_cli_path(args.data_dir),
                expected_data_sha256=args.expected_data_sha256,
                cancel_event=cancel_event,
            )
        payload = _result_manifest(result, location="closed-bundle replay")
    except (CanonicalDataError, OrganizerIntegrityError, OSError, RuntimeError) as exc:
        print(f"CAMPAIGN_REPLAY_FAILED: {exc}", file=sys.stderr)
        return EXIT_CONTRACT
    print(_stable_json(payload))
    return 0


def _resolved_from(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    return candidate.resolve(strict=False)


def _run(args: argparse.Namespace) -> int:
    """Create, drive, and deterministically finalize one new qualified campaign."""

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"CONFIG_INVALID: {exc}", file=sys.stderr)
        return EXIT_INVALID
    try:
        repository_root = Path.cwd().resolve(strict=True)
        data_dir = _resolved_from(repository_root, config.benchmark.data_dir)
        dataset = load_canonical_dataset(data_dir)
        if dataset.final.targets is not None:
            raise CanonicalDataError("final split unexpectedly exposes a target capability")
        run_dir_arg: Path | None = args.run_dir
        run_dir = (
            _resolved_from(repository_root, run_dir_arg)
            if run_dir_arg is not None
            else repository_root / DEFAULT_CAMPAIGN_ROOT / f"campaign-{config.digest[:12]}"
        )
        qualification_run_dir = _resolved_from(
            repository_root,
            args.qualification_run_dir,
        )
        provenance = build_campaign_request(
            repository_root=repository_root,
            run_dir=run_dir,
            qualification_run_dir=qualification_run_dir,
            config=config,
            dataset_manifest_digest=dataset.digest,
        )
        engine = CampaignEngine()
        engine.create(provenance.request)
    except (
        CampaignControllerError,
        CanonicalDataError,
        OrganizerIntegrityError,
        ProvenanceError,
        OSError,
        RuntimeError,
    ) as exc:
        print(f"CAMPAIGN_CREATE_FAILED: {exc}", file=sys.stderr)
        return EXIT_CONTRACT
    try:
        with cancellation_on_signals() as cancel_event:
            _drive_provider_free_campaign(
                run_dir,
                project_root=repository_root,
                engine=engine,
                cancel_event=cancel_event,
            )
            finalized = _finalize_provider_free_campaign(
                run_dir,
                project_root=repository_root,
                engine=engine,
                cancel_event=cancel_event,
            )
        payload = _result_manifest(finalized, location="campaign finalization")
    except (CampaignControllerError, OSError, RuntimeError) as exc:
        print(f"CAMPAIGN_RUN_FAILED: {exc}", file=sys.stderr)
        return EXIT_CONTRACT
    print(_stable_json(payload))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the complete stable command surface."""

    parser = argparse.ArgumentParser(
        prog="kuairand-agent",
        description="Leakage-safe autonomous ML research for KuaiRand-Pure",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    data_parser = commands.add_parser("data", help="prepare or audit benchmark data")
    data_commands = data_parser.add_subparsers(dest="data_command", required=True)
    prepare = data_commands.add_parser("prepare", help="verify and securely prepare an archive")
    source = prepare.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--download", action="store_true")
    prepare.add_argument("--data-dir", required=True, type=Path)
    prepare.set_defaults(handler=_prepare_data, command_path=("data", "prepare"))
    audit = data_commands.add_parser("audit", help="audit a prepared dataset without final labels")
    audit.add_argument("--data-dir", required=True, type=Path)
    audit.add_argument("--json", action="store_true")
    audit.add_argument("--output-dir", type=Path, help="write paired JSON and Markdown evidence")
    audit.set_defaults(handler=_audit_data, command_path=("data", "audit"))

    qualify = commands.add_parser("qualify", help="reproduce organizer baselines and fallback")
    qualify.add_argument("--data-dir", required=True, type=Path)
    qualify.add_argument("--run-dir", required=True, type=Path)
    qualify.add_argument("--starter-dir", type=Path, default=Path("kuairand-starter-kit"))
    qualify.set_defaults(handler=_qualify, command_path=("qualify",))

    run = commands.add_parser("run", help="create and drive a new campaign")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument(
        "--qualification-run-dir",
        type=Path,
        default=DEFAULT_QUALIFICATION_RUN_DIR,
        help="qualified official-FM evidence (default: runs/wp3-official-qualification)",
    )
    run.add_argument(
        "--run-dir",
        type=Path,
        help="new campaign path (default: runs/campaign-<config digest>)",
    )
    run.set_defaults(handler=_run, command_path=("run",))

    resume = commands.add_parser("resume", help="reconcile and continue a campaign")
    resume.add_argument("--run-dir", required=True, type=Path)
    resume.set_defaults(handler=_resume, command_path=("resume",))

    status = commands.add_parser("status", help="read campaign status without mutation")
    status.add_argument("--run-dir", required=True, type=Path)
    status.add_argument("--json", action="store_true")
    status.set_defaults(handler=_status, command_path=("status",))

    iteration_log = commands.add_parser(
        "iteration-log", help="emit the per-iteration run log required by the starter kit"
    )
    iteration_log.add_argument("--run-dir", required=True, type=Path)
    iteration_log.add_argument("--format", choices=("md", "jsonl"), default="md")
    iteration_log.add_argument("--output", type=Path)
    iteration_log.set_defaults(handler=_iteration_log, command_path=("iteration-log",))

    finalize = commands.add_parser("finalize", help="stop research and finalize deterministically")
    finalize.add_argument("--run-dir", required=True, type=Path)
    finalize.set_defaults(handler=_finalize, command_path=("finalize",))

    replay = commands.add_parser("replay", help="replay a frozen final bundle without a provider")
    replay.add_argument("--bundle", required=True, type=Path)
    replay.add_argument("--project-root", required=True, type=Path)
    replay.add_argument("--data-dir", required=True, type=Path)
    replay.add_argument("--expected-data-sha256", required=True, type=_sha256_argument)
    replay.set_defaults(handler=_replay, command_path=("replay",))

    validate = commands.add_parser(
        "validate-submission", help="validate a high-precision aligned submission"
    )
    validate.add_argument("--split", required=True, choices=("valid", "test"))
    validate.add_argument("file", type=Path)
    validate.add_argument("--data-dir", type=Path)
    validate.set_defaults(handler=_validate_submission, command_path=("validate-submission",))

    config_parser = commands.add_parser("config", help="validate and normalize configuration")
    config_commands = config_parser.add_subparsers(dest="config_command", required=True)
    config_validate = config_commands.add_parser("validate", help="validate one TOML configuration")
    config_validate.add_argument("file", type=Path)
    config_validate.set_defaults(handler=_validate_config, command_path=("config", "validate"))

    provider_parser = commands.add_parser(
        "provider", help="check provider availability without making a research call"
    )
    provider_commands = provider_parser.add_subparsers(dest="provider_command", required=True)
    provider_preflight = provider_commands.add_parser(
        "preflight", help="validate provider configuration and credential availability"
    )
    provider_preflight.add_argument("--config", required=True, type=Path)
    provider_preflight.set_defaults(
        handler=_preflight_provider,
        command_path=("provider", "preflight"),
    )

    contract_parser = commands.add_parser("contract", help="verify immutable organizer artifacts")
    contract_commands = contract_parser.add_subparsers(dest="contract_command", required=True)
    starter = contract_commands.add_parser("verify-starter", help="verify all starter members")
    starter.add_argument("--starter-dir", type=Path, default=Path("kuairand-starter-kit"))
    starter.set_defaults(handler=_verify_starter, command_path=("contract", "verify-starter"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and return a stable process exit code."""

    args = build_parser().parse_args(argv)
    handler = getattr(args, "handler", None)
    if not callable(handler):  # defensive; required subparsers should make this unreachable
        print("INTERNAL_ERROR: command has no handler", file=sys.stderr)
        return 1
    return int(handler(args))


def entrypoint() -> None:
    """Console-script entry point."""

    raise SystemExit(main())
