from app.routes.api import router as api_router
from app.routes.docs import router as docs_router
from app.routes.docs import swagger_ui_blueprint as swagger_router

routers = [api_router, docs_router, swagger_router]