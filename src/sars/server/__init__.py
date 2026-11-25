"""sars.server: the library's operations over HTTP.

Install with ``pip install "sars[server]"`` and start with ``sars serve``. Each
endpoint runs one operation, the same ones the command line runs, and returns
its JSON envelope.
"""

from sars._extras import require_extra

require_extra("server", "sars.server")

from sars.server.app import (  # noqa: E402
    API_VERSION,
    ServerConfig,
    create_app,
    openapi_spec,
    write_openapi_spec,
)

__all__ = ["API_VERSION", "ServerConfig", "create_app", "openapi_spec", "write_openapi_spec"]
