"""Keep multipart uploads in RAM; SecurityMiddleware bounds request size."""
from fastapi.routing import APIRoute
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.exceptions import HTTPException


class MemoryMultipartParser(MultiPartParser):
    # SpooledTemporaryFile with max_size=0 never automatically rolls to disk.
    spool_max_size = 0
    # Compatibility with earlier supported Starlette versions.
    max_file_size = 0


class MemoryUploadRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def memory_handler(request):
            form = None
            if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() == "multipart/form-data":
                try:
                    form = await MemoryMultipartParser(
                        request.headers, request.stream(), max_files=1, max_fields=0
                    ).parse()
                except MultiPartException:
                    raise HTTPException(400, "Upload harus berisi satu gambar") from None
                # Seed the same per-request form cache consumed by FastAPI.
                request._form = form
            try:
                return await handler(request)
            finally:
                if form is not None:
                    await form.close()

        return memory_handler
