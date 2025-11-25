import argparse
import yaml
import sys
from pathlib import Path

from nyuctf.dataset import CTFDataset
from nyuctf.challenge import CTFChallenge

from nyuctf_multiagent.environment import CTFEnvironment
from nyuctf_multiagent.backends import MODELS, Role
from nyuctf_multiagent.prompting import PromptManager
from nyuctf_multiagent.agent import PlannerExecutorSystem, PlannerAgent, ExecutorAgent, AutoPromptAgent
from nyuctf_multiagent.logging import logger
from nyuctf_multiagent.utils import APIKeys, load_common_options, get_log_filename, load_config
from nyuctf_multiagent.config import Config

parser = argparse.ArgumentParser(description="Multi-agent Planner-Executor LLM for CTF solving")

# Loads the dataset and container related common options into parser
load_common_options(parser)
parser.add_argument("--logdir", default="logs_dcipher", type=str, help="Log directory")
parser.add_argument("--config", default=None, help="YAML config for the planner-executor multiagent. If not provided, it picks one automatically based on challenge cateogory.")

# Config overriding options
parser.add_argument("--planner-model", default=None, help="Planner model to use (overrides config)")
parser.add_argument("--executor-model", default=None, help="Executor model to use (overrides config)")
parser.add_argument("--autoprompter-model", default=None, help="AutoPrompt model to use (overrides config)")
parser.add_argument("--max-cost", default=0.0, type=float, help="Max cost in $ (overrides config)")
parser.add_argument("--enable-autoprompt", action="store_true", help="Init prompt message auto generated, else use generic base prompt")
parser.add_argument("--strict", action="store_true", help="Enable strict mode for function calling") # TODO only works for Together, add to other backends

parser.add_argument("--planner_temperature", default=None, type=float, help="Temperature for the planner_model")
parser.add_argument("--planner_top_p", default=None, type=float, help="Top_p for the planner_model")
parser.add_argument("--executor_temperature", default=None, type=float, help="Temperature for the executor_model")
parser.add_argument("--executor_top_p", default=None, type=float, help="Top_p for the executor_model")
parser.add_argument("--planner_token", default=None, type=int, help="max_tokens")

parser.add_argument("--executor_token", default=None, type=int, help="max_tokens")

args = parser.parse_args()

logger.set(quiet=args.quiet, debug=args.debug)

if args.dataset is not None:
    dataset = CTFDataset(dataset_json=args.dataset)
else:
    dataset = CTFDataset(split=args.split)
challenge = CTFChallenge(dataset.get(args.challenge), dataset.basedir)
logfile = get_log_filename(args, challenge)

logger.print(f"Logging to {str(logfile)}", force=True)
if logfile.exists() and args.skip_existing:
    logger.print("Skipping as log file exists", force=True)
    exit(0)

keys = APIKeys(args.keys)
environment = CTFEnvironment(challenge, args.container_image, args.container_network)

if args.config:
    config_f = Path(args.config)
else:
    config_d = Path(sys.argv[0]).parent / "configs" / "dcipher"
    config_f = config_d / f"{challenge.category}_planner_executor.yaml"

logger.print(f"Using config: {str(config_f)}", force=True)
config = load_config(config_f, args=args)

base_url = getattr(config.experiment, "base_url", None)

config.experiment.enable_autoprompt = True if args.enable_autoprompt else config.experiment.enable_autoprompt
if args.strict:
    config.planner.strict = True
    config.executor.strict = True
    config.autoprompter.strict = True

if args.executor_temperature is not None:
    config.planner.temperature = args.planner_temperature
    config.executor.temperature = args.executor_temperature
    #config.autoprompter.temperature = args.temperature

if args.executor_top_p:
    config.planner.top_p = args.planner_top_p
    config.executor.top_p = args.executor_top_p
    #config.autoprompter.top_p = args.top_p
if args.executor_token:
    config.planner.max_tokens = args.planner_token
    config.executor.max_tokens = args.executor_token
    print("args.executor_token", args.executor_token)
    print("args.executor_top_p", args.executor_top_p)
    print("args.executor_temperature", args.executor_temperature)
    #config.autoprompter.top_p = args.top_p
autoprompter_backend_cls = MODELS[config.autoprompter.model]
autoprompter_backend_kwargs = {
    "role": Role.AUTOPROMPTER,
    "model": config.autoprompter.model,
    "tools": environment.get_toolset(config.autoprompter.toolset),
    "api_key": keys.get(autoprompter_backend_cls.NAME.upper()),
    "config": config,
}
if autoprompter_backend_cls.NAME == "openai":
    autoprompter_backend_kwargs["base_url"] = base_url
    if base_url:
        autoprompter_backend_kwargs["api_key"] = "NA"
autoprompter_backend = autoprompter_backend_cls(**autoprompter_backend_kwargs)
autoprompter_prompter = PromptManager(config_f.parent / config.autoprompter.prompt, challenge, environment)
autoprompter = AutoPromptAgent(environment, challenge, autoprompter_prompter,
                       autoprompter_backend, max_rounds=config.autoprompter.max_rounds)

if config.experiment.enable_autoprompt:
    autoprompter.enable_autoprompt()

planner_backend_cls = MODELS[config.planner.model]
planner_backend_kwargs = {
    "role": Role.PLANNER,
    "model": config.planner.model,
    "tools": environment.get_toolset(config.planner.toolset),
    "api_key": keys.get(planner_backend_cls.NAME.upper()),
    "config": config,
}
if planner_backend_cls.NAME == "openai":
    planner_backend_kwargs["base_url"] = base_url
    if base_url:
        planner_backend_kwargs["api_key"] = "NA"
planner_backend = planner_backend_cls(**planner_backend_kwargs)
planner_prompter = PromptManager(config_f.parent / config.planner.prompt, challenge, environment)
planner = PlannerAgent(environment, challenge, planner_prompter,
                       planner_backend, max_rounds=config.planner.max_rounds)

executor_backend_cls = MODELS[config.executor.model]
executor_backend_kwargs = {
    "role": Role.EXECUTOR,
    "model": config.executor.model,
    "tools": environment.get_toolset(config.executor.toolset),
    "api_key": keys.get(executor_backend_cls.NAME.upper()),
    "config": config,
}
if executor_backend_cls.NAME == "openai":
    executor_backend_kwargs["base_url"] = base_url
    if base_url:
        executor_backend_kwargs["api_key"] = "NA"
executor_backend = executor_backend_cls(**executor_backend_kwargs)
executor_prompter = PromptManager(config_f.parent / config.executor.prompt, challenge, environment)
executor = ExecutorAgent(environment, challenge, executor_prompter,
                         executor_backend, max_rounds=config.executor.max_rounds)
executor.conversation.len_observations = config.executor.len_observations

with PlannerExecutorSystem(environment, challenge, autoprompter, planner, executor, max_cost=config.experiment.max_cost, logfile=logfile) as multiagent:
    multiagent.run()
