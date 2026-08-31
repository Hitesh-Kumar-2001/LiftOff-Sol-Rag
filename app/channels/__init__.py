"""Messaging gateways: reaching this service from WhatsApp, LINE, and the rest.

``channel`` is the ``Channel`` protocol every gateway satisfies -- verify a
delivery, parse it, send a reply. ``registry`` is the one place a gateway is
registered. ``sender`` is the single outbound path the routes call. One module
per platform beside them.

The routes are in ``app.api.channelRoutes`` and the per-project credentials in
``app.stores.channelStore``.
"""
