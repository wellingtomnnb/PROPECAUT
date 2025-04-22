function im_out = adjust_image(im_in)

im_aux = im_in-min(min(im_in));
im_out = uint8(255*(im_aux/(max(max(im_aux)))));

end
