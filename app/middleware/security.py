from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware that appends security hardening HTTP response headers
    as per OWASP and HIPAA audit guidelines.
    """
    async def dispatch(self, request: Request, call_next) -> Response:
        response: Response = await call_next(request)
        
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        
        # Force browser to strictly follow Content-Type headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Cross-Site Scripting protection for legacy browsers
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Enforce HTTPS (HTTP Strict Transport Security)
        # Note: Set max-age to 1 year (31536000 seconds)
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Content Security Policy - restricts resource loading to trusted sources
        # response.headers["Content-Security-Policy"] = (
        #     # "default-src 'self'; "
        #     # "script-src 'self' 'unsafe-inline'; "
        #     # "style-src 'self' 'unsafe-inline'; "
        #     # "img-src 'self' data:; "
        #     # "connect-src 'self' http://localhost:8000 ws://localhost:8000;"
        #     "default-src 'self'; "
        #     "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        #     "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        #     "font-src 'self' https://fonts.gstatic.com; "
        #     "img-src 'self' data: https://fastapi.tiangolo.com;"
        # )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            # Map tiles are fetched by Leaflet straight from OpenStreetMap.
            # Without this host the map renders as an empty grey grid. Marker
            # icons are inline SVG data URIs, already covered by `data:`.
            "img-src 'self' data: https://fastapi.tiangolo.com "
            "https://*.tile.openstreetmap.org; "
            "connect-src 'self' ws: wss:;"
        )
                
        # Permissions Policy - allow geolocation for Emergency SOS feature
        response.headers["Permissions-Policy"] = "geolocation=(self), microphone=(), camera=()"
        
        # Cross-Origin Security Headers
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "unsafe-none"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"

        return response
