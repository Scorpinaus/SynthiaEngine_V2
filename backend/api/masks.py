"""Mask image utility endpoints."""

from io import BytesIO

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from PIL import Image, ImageFilter


router = APIRouter(tags=["masks"])


def create_blur_mask(mask_image: Image.Image, blur_factor: int) -> Image.Image:
    """Apply a Gaussian blur using the existing clamped radius contract."""
    clamped_blur = max(0, min(int(blur_factor), 128))
    if clamped_blur == 0:
        return mask_image
    return mask_image.filter(ImageFilter.GaussianBlur(radius=clamped_blur))


@router.post("/create-blur-mask")
async def create_blur_mask_endpoint(
    mask_image: UploadFile = File(...),
    blur_factor: int = Form(8),
):
    """Generate a blurred grayscale mask used by inpainting pages."""
    mask_bytes = await mask_image.read()
    try:
        mask = Image.open(BytesIO(mask_bytes)).convert("L")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid mask image file.") from exc

    blurred_mask = create_blur_mask(mask, blur_factor)
    output = BytesIO()
    blurred_mask.save(output, format="PNG")
    return Response(content=output.getvalue(), media_type="image/png")
