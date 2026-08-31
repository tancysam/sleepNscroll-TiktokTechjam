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
from kuairand_agent.scoring.submission import AlignmentRow, SubmissionError, read_submission

EXIT_INVALID: Final = 2
EXIT_CONTRACT: Final = 3
DEFAULT_QUALIFICATION_RUN_DIR: Final = Path("runs/wp3-official-qualification")
LEGACY_MUTATION_ERROR: Final = (
    "run-directory campaign mutation is disabled because it bypasses the StateRepository "
    "authority; start or retry work with 'kuairand-agent compete --config CONFIG "
    "--state-root STATE_ROOT --run-root RUN_ROOT', then use 'inspect' or 'replay' with the "
    "returned campaign id"
)


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


def _compete(args: argparse.Namespace) -> int:
    """Enter the production-shaped facade and fail closed on missing admission evidence."""

    from kuairand_agent.lab import (
        AutonomousExperimentLab,
        CampaignOptions,
        LabError,
    )
    from kuairand_agent.resource_profiles import ResourceProfileError, load_resource_profile

    repository_root = Path.cwd().resolve(strict=True)
    config_path = _absolute_cli_path(args.config)
    try:
        profile = load_resource_profile(config_path)
        requested_profile = args.profile or profile.name
        lab = AutonomousExperimentLab.open(
            repository_root=repository_root,
            state_root=_absolute_cli_path(args.state_root),
            run_root=_absolute_cli_path(args.run_root),
            profile=requested_profile,
        )
        key = args.idempotency_key or f"cli:{_absolute_cli_path(args.run_root)}"
        result = lab.compete(
            options=CampaignOptions(config_path=config_path),
            idempotency_key=key,
        )
    except (LabError, ResourceProfileError, OSError, RuntimeError) as exc:
        print(f"LAB_COMPETE_FAILED: {exc}", file=sys.stderr)
        return EXIT_CONTRACT
    print(_stable_json(result.manifest()))
    return 0


def _inspect_lab(args: argparse.Namespace) -> int:
    """Inspect only the SQLite authority; never reconcile or rebuild projections."""

    from kuairand_agent.lab import AutonomousExperimentLab, LabError
    from kuairand_agent.state.projections import ProjectionError
    from kuairand_agent.state.repository import StateError

    try:
        repository_root = Path.cwd().resolve(strict=True)
        lab = AutonomousExperimentLab.open(
            repository_root=repository_root,
            state_root=_absolute_cli_path(args.state_root),
            run_root=repository_root / "runs",
            profile="cpu",
        )
        snapshot = lab.inspect(campaign_id=args.campaign_id)
    except (LabError, ProjectionError, StateError, OSError, RuntimeError) as exc:
        print(f"LAB_INSPECT_FAILED: {exc}", file=sys.stderr)
        return EXIT_CONTRACT
    if args.json:
        print(_stable_json(snapshot))
    else:
        campaign = snapshot.get("campaign")
        if not isinstance(campaign, Mapping):
            print("LAB_INSPECT_FAILED: authority projection has no campaign", file=sys.stderr)
            return EXIT_CONTRACT
        print(
            f"{campaign.get('campaign_id')}: {campaign.get('state')}; "
            f"revision {campaign.get('revision')}"
        )
    return 0


def _validate_lab_bundle(args: argparse.Namespace) -> int:
    """Validate one sealed bundle without opening or mutating campaign state."""

    from kuairand_agent.lab import AutonomousExperimentLab, LabError

    try:
        result = AutonomousExperimentLab.validate_bundle(_absolute_cli_path(args.bundle))
    except (LabError, OSError, RuntimeError) as exc:
        print(f"BUNDLE_VALIDATION_FAILED: {exc}", file=sys.stderr)
        return EXIT_CONTRACT
    print(_stable_json(result.manifest()))
    return 0


def _replay_lab(args: argparse.Namespace) -> int:
    """Verify a named replay grade from the new authority and exact bundle."""

    from kuairand_agent.lab import AutonomousExperimentLab, LabError
    from kuairand_agent.state.projections import ProjectionError
    from kuairand_agent.state.repository import StateError

    try:
        repository_root = Path.cwd().resolve(strict=True)
        lab = AutonomousExperimentLab.open(
            repository_root=repository_root,
            state_root=_absolute_cli_path(args.state_root),
            run_root=repository_root / "runs",
            profile="cpu",
        )
        result = lab.replay(campaign_id=args.campaign_id, grade=args.grade)
    except (LabError, ProjectionError, StateError, OSError, RuntimeError) as exc:
        print(f"LAB_REPLAY_FAILED: {exc}", file=sys.stderr)
        return EXIT_CONTRACT
    print(_stable_json(result.manifest()))
    return 0


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


def _resume(args: argparse.Namespace) -> int:
    """Fail closed rather than resume through the retired run-directory authority."""

    del args
    print(f"LEGACY_CAMPAIGN_COMMAND_DISABLED: resume: {LEGACY_MUTATION_ERROR}", file=sys.stderr)
    return EXIT_CONTRACT


def _finalize(args: argparse.Namespace) -> int:
    """Fail closed rather than finalize through the retired run-directory authority."""

    del args
    print(f"LEGACY_CAMPAIGN_COMMAND_DISABLED: finalize: {LEGACY_MUTATION_ERROR}", file=sys.stderr)
    return EXIT_CONTRACT


def _replay(args: argparse.Namespace) -> int:
    """Replay a closed SHA-bound bundle with canonical label-free capabilities."""

    if getattr(args, "campaign_id", None) is not None:
        return _replay_lab(args)
    if (
        args.bundle is None
        or args.project_root is None
        or args.data_dir is None
        or args.expected_data_sha256 is None
    ):
        print(
            "CAMPAIGN_REPLAY_FAILED: legacy bundle replay requires --bundle, "
            "--project-root, --data-dir, and --expected-data-sha256",
            file=sys.stderr,
        )
        return EXIT_INVALID

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


def _run(args: argparse.Namespace) -> int:
    """Fail closed rather than create state in the retired run-directory authority."""

    del args
    print(f"LEGACY_CAMPAIGN_COMMAND_DISABLED: run: {LEGACY_MUTATION_ERROR}", file=sys.stderr)
    return EXIT_CONTRACT


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

    compete = commands.add_parser(
        "compete",
        help="enter the autonomous laboratory's production admission gate",
    )
    compete.add_argument("--config", required=True, type=Path)
    compete.add_argument("--state-root", type=Path, default=Path(".kuairand"))
    compete.add_argument("--run-root", required=True, type=Path)
    compete.add_argument(
        "--profile",
        choices=("cpu", "gpu", "competition-cpu", "competition-gpu"),
        help="optional explicit check against the profile named by --config",
    )
    compete.add_argument(
        "--idempotency-key",
        help="stable retry key (default: the absolute --run-root path)",
    )
    compete.set_defaults(handler=_compete, command_path=("compete",))

    inspect = commands.add_parser(
        "inspect",
        help="read a laboratory campaign from the SQLite authority without mutation",
    )
    inspect.add_argument("--state-root", type=Path, default=Path(".kuairand"))
    inspect.add_argument("--campaign-id", required=True, type=_sha256_argument)
    inspect.add_argument("--json", action="store_true")
    inspect.set_defaults(handler=_inspect_lab, command_path=("inspect",))

    run = commands.add_parser(
        "run",
        help="legacy parser compatibility only; use compete for StateRepository campaigns",
    )
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

    resume = commands.add_parser(
        "resume",
        help="legacy parser compatibility only; retry the original compete command",
    )
    resume.add_argument("--run-dir", required=True, type=Path)
    resume.set_defaults(handler=_resume, command_path=("resume",))

    status = commands.add_parser("status", help="read campaign status without mutation")
    status.add_argument("--run-dir", required=True, type=Path)
    status.add_argument("--json", action="store_true")
    status.set_defaults(handler=_status, command_path=("status",))

    finalize = commands.add_parser(
        "finalize",
        help="legacy parser compatibility only; compete owns atomic finalization",
    )
    finalize.add_argument("--run-dir", required=True, type=Path)
    finalize.set_defaults(handler=_finalize, command_path=("finalize",))

    replay = commands.add_parser(
        "replay",
        help="replay either a legacy bundle or a campaign in the new laboratory authority",
    )
    replay_source = replay.add_mutually_exclusive_group(required=True)
    replay_source.add_argument("--bundle", type=Path)
    replay_source.add_argument("--campaign-id", type=_sha256_argument)
    replay.add_argument("--project-root", type=Path)
    replay.add_argument("--data-dir", type=Path)
    replay.add_argument("--expected-data-sha256", type=_sha256_argument)
    replay.add_argument("--state-root", type=Path, default=Path(".kuairand"))
    replay.add_argument(
        "--grade",
        choices=(
            "experiment-same-backend",
            "scoring-exact",
            "bundle-exact",
            "EXPERIMENT_SAME_BACKEND",
            "SCORING_EXACT",
            "BUNDLE_EXACT",
        ),
        default="experiment-same-backend",
    )
    replay.set_defaults(handler=_replay, command_path=("replay",))

    validate_bundle = commands.add_parser(
        "validate-bundle",
        help="verify exact bundle membership, hashes, and BundleId without state writes",
    )
    validate_bundle.add_argument("--bundle", required=True, type=Path)
    validate_bundle.set_defaults(
        handler=_validate_lab_bundle,
        command_path=("validate-bundle",),
    )

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
