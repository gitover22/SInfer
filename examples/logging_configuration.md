# Logging Configuration

SInfer leverages Python's `logging.config.dictConfig` functionality to enable
robust and flexible configuration of the various loggers used by SInfer.

SInfer offers two environment variables that can be used to accommodate a range
of logging configurations that range from simple-and-inflexible to
more-complex-and-more-flexible.

- No SInfer logging (simple and inflexible)
  - Set `SINFER_CONFIGURE_LOGGING=0` (leaving `SINFER_LOGGING_CONFIG_PATH` unset)
- SInfer's default logging configuration (simple and inflexible)
  - Leave `SINFER_CONFIGURE_LOGGING` unset or set `SINFER_CONFIGURE_LOGGING=1`
- Fine-grained custom logging configuration (more complex, more flexible)
  - Leave `SINFER_CONFIGURE_LOGGING` unset or set `SINFER_CONFIGURE_LOGGING=1` and
    set `SINFER_LOGGING_CONFIG_PATH=<path-to-logging-config.json>`


## Logging Configuration Environment Variables

### `SINFER_CONFIGURE_LOGGING`

`SINFER_CONFIGURE_LOGGING` controls whether or not SInfer takes any action to
configure the loggers used by SInfer. This functionality is enabled by default,
but can be disabled by setting `SINFER_CONFIGURE_LOGGING=0` when running SInfer.

If `SINFER_CONFIGURE_LOGGING` is enabled and no value is given for
`SINFER_LOGGING_CONFIG_PATH`, SInfer will use built-in default configuration to
configure the root SInfer logger. By default, no other SInfer loggers are
configured and, as such, all SInfer loggers defer to the root SInfer logger to make
all logging decisions.

If `SINFER_CONFIGURE_LOGGING` is disabled and a value is given for
`SINFER_LOGGING_CONFIG_PATH`, an error will occur while starting SInfer.

### `SINFER_LOGGING_CONFIG_PATH`

`SINFER_LOGGING_CONFIG_PATH` allows users to specify a path to a JSON file of
alternative, custom logging configuration that will be used instead of SInfer's
built-in default logging configuration. The logging configuration should be
provided in JSON format following the schema specified by Python's [logging
configuration dictionary
schema](https://docs.python.org/3/library/logging.config.html#dictionary-schema-details).

If `SINFER_LOGGING_CONFIG_PATH` is specified, but `SINFER_CONFIGURE_LOGGING` is
disabled, an error will occur while starting SInfer.


## Examples

### Example 1: Customize SInfer root logger

For this example, we will customize the SInfer root logger to use
[`python-json-logger`](https://github.com/madzak/python-json-logger) to log to
STDOUT of the console in JSON format with a log level of `INFO`.

To begin, first, create an appropriate JSON logging configuration file:

**/path/to/logging_config.json:**

```json
{
  "formatters": {
    "json": {
      "class": "pythonjsonlogger.jsonlogger.JsonFormatter"
    }
  },
  "handlers": {
    "console": {
      "class" : "logging.StreamHandler",
      "formatter": "json",
      "level": "INFO",
      "stream": "ext://sys.stdout"
    }
  },
  "loggers": {
    "sinfer": {
      "handlers": ["console"],
      "level": "INFO",
      "propagate": false
    }
  },
  "version": 1
}
```

Next, install the `python-json-logger` package if it's not already installed:

```bash
pip install python-json-logger
```

Finally, run SInfer with the `SINFER_LOGGING_CONFIG_PATH` environment variable set
to the path of the custom logging configuration JSON file:

```bash
SINFER_LOGGING_CONFIG_PATH=/path/to/logging_config.json \
    python3 -m sinfer.entrypoints.openai.api_server \
    --max-model-len 2048 \
    --model mistralai/Mistral-7B-v0.1
```


### Example 2: Silence a particular SInfer logger

To silence a particular SInfer logger, it is necessary to provide custom logging
configuration for the target logger that configures the logger so that it won't
propagate its log messages to the root SInfer logger.

When custom configuration is provided for any logger, it is also necessary to
provide configuration for the root SInfer logger since any custom logger
configuration overrides the built-in default logging configuration used by SInfer.

First, create an appropriate JSON logging configuration file that includes
configuration for the root SInfer logger and for the logger you wish to silence:

**/path/to/logging_config.json:**

```json
{
  "formatters": {
    "sinfer": {
      "class": "sinfer.logging.NewLineFormatter",
      "datefmt": "%m-%d %H:%M:%S",
      "format": "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
    }
  },
  "handlers": {
    "sinfer": {
      "class" : "logging.StreamHandler",
      "formatter": "sinfer",
      "level": "INFO",
      "stream": "ext://sys.stdout"
    }
  },
  "loggers": {
    "sinfer": {
      "handlers": ["sinfer"],
      "level": "DEBUG",
      "propagage": false
    },
    "sinfer.example_noisy_logger": {
      "propagate": false
    }
  },
  "version": 1
}
```

Finally, run SInfer with the `SINFER_LOGGING_CONFIG_PATH` environment variable set
to the path of the custom logging configuration JSON file:

```bash
SINFER_LOGGING_CONFIG_PATH=/path/to/logging_config.json \
    python3 -m sinfer.entrypoints.openai.api_server \
    --max-model-len 2048 \
    --model mistralai/Mistral-7B-v0.1
```


### Example 3: Disable SInfer default logging configuration

To disable SInfer's default logging configuration and silence all SInfer loggers,
simple set `SINFER_CONFIGURE_LOGGING=0` when running SInfer. This will prevent SInfer
for configuring the root SInfer logger, which in turn, silences all other SInfer
loggers.

```bash
SINFER_CONFIGURE_LOGGING=0 \
    python3 -m sinfer.entrypoints.openai.api_server \
    --max-model-len 2048 \
    --model mistralai/Mistral-7B-v0.1
```


## Additional resources

- [`logging.config` Dictionary Schema Details](https://docs.python.org/3/library/logging.config.html#dictionary-schema-details)
