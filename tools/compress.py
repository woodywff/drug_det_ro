#################################
# Borrowed From: Someone Online #
#################################

from PIL import Image
import os

def get_size(file):
    '''
    Get file size in KB
    '''
    size = os.path.getsize(file)
    return round(size / 1024 / 1024, 3)

def compress(infile, outfile, mb=1.5, step=10, quality=80):
    '''
    Compress the image into a smaller storage without changing the image shape.
    infile: input image path
    outfile: save path
    mb: target storage size in MB
    step: changing step for save quality
    quality: param for Image.save
    '''
    o_size = get_size(infile)
    if o_size <= mb:
        im = Image.open(infile).convert("RGB")
        im.save(outfile, quality=quality)
        return infile
    while o_size > mb:
        im = Image.open(infile).convert("RGB")
        im.save(outfile, quality=quality)
        if quality - step < 0:
            break
        quality -= step
        o_size = get_size(outfile)
    return outfile, get_size(outfile)