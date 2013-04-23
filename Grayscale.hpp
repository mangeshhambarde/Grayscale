
/* Grayscale.hpp
 * Mangesh Hambarde
 * github.com/rttlesnke
 * 21st April, 2013
*/

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include "jpeg/jpeglib.h"

class Grayscale
{
    private:
        std::vector<uint8_t>  image;
        int                   width;
        int                   height;
        int                   pixels;
        bool                  valid;

    public:
        /* Basic operations */
        Grayscale(const std::string& src);                                              // Loads JPG
        Grayscale(const std::string& src, int width, int height);                       // Loads RAW
        Grayscale(int width, int height);                                               // Temporary image
        ~Grayscale();                                                                   // Destructor
        void                cleanup();                                                  // Frees memory, marks state as invalid
        bool                is_valid();                                                 // Check if object is valid
        int                 write_raw(const std::string& des);                          // Writes RAW file to des
        int                 write_jpg(const std::string& des);                          // Writes JPG file to des
        int                 get_width();                                                // Returns width
        int                 get_height();                                               // Returns height
        int                 get_pixels();                                               // Returns number of pixels
        inline void         get_pixel(uint8_t *color, int i, int j);                    // Returns color of pixel (i,j) in *color
        inline void         set_pixel(uint8_t color, int i, int j);                     // Sets color of pixel (i,j) to color
        inline uint8_t&     operator()(int i);                                          // Returns reference to pixel, image visualized as 1D array
        inline uint8_t&     operator()(int i, int j);                                   // Returns reference to pixel at (i,j)

        /* Common manipulations */
        void                rotate(uint8_t dir);                                        // Rotates image L or R
        void                horizontal_flip();                                          // Flip horizontally
        void                vertical_flip();                                            // Flip vertically
        void                resize(int width, int height);                              // Resizes image (nearest neighbour)
        void                pad(int padding, uint8_t value);                            // Pads image by value
		void				pad_horizontal(int padding, uint8_t value);
		void				pad_vertical(int padding, uint8_t value);
        void                save_hist(const std::string& des);                          // Save JPG histogram to disk
		void				draw_line(double m, double c);

        /* Pixel operations */
        void                threshold(uint8_t t);                                       // Threshold image
        void                gamma_correction(double gamma);                             // Gamma correction
        void                contrast_stretch(int r1, int s1, int r2, int s2);           // Stretches range intensity
        void                bit_plane_slice(uint8_t i);                                 // Bit plane slicing
        void                negative();                                                 // Negates image
        void                hist_eq();                                                  // Histogram equalization
        void                hist_spec(const std::string& src);                          // Histogram specification with reference JPG file

        /* Linear spacial filters */
        void                conv(float* kernel, int n);                                 // Convolves image with kernel
        void                mean_filter(int n);                                         // Mean smoothing filter
        void                unsharp_filter(int kernel_size, float weight);              // Sharpens image with unsharp masking

        /* Non-linear spacial filters */
        void                median_filter(int n);                                       // Median filter
        void                max_neighbour(int n);                                       // Maximum pixel in neighbourhood
        void                min_neighbour(int n);                                       // Minimum pixel in neighbourhood
        void                heat_map();                                                 // Converts to a heat map

        /* Edge detection */
        void                edge_sobel();                                               // Applies the Sobel operator
        void                edge_prewitt();                                             // Applies the Prewitt operator

        /* Morphological operations */
		void				dilate(bool* se, int s_width, int s_height);				// Dilation
		void				erode(bool* se, int s_width, int s_height);					// Erosion
		void				open(bool* se, int s_width, int s_height);					// Morphological opening
		void				close(bool* se, int s_width, int s_height);					// Morphological closing
		void				boundary(bool* se, int s_width, int s_height);				// Extract boundary
		void				hough(int threshold);										// Hough transform with threshold
};

Grayscale :: Grayscale(const std::string& src)
{
    // Check extension
    int len = src.length();
    if( 
		!(toupper(src[len-3])=='J' && toupper(src[len-2])=='P' && toupper(src[len-1])=='G') &&
		!(toupper(src[len-4])=='J' && toupper(src[len-3])=='P' && toupper(src[len-2])=='E' && toupper(src[len-1])=='G') )
    {
        cleanup();
        fprintf(stderr, "Extension not valid, assuming not a JPG: %s\n", src.c_str());
        return;
    }

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE *in;
    JSAMPARRAY buffer;

    if( (in = fopen(src.c_str(), "rb")) == NULL )
    {
        cleanup();
        fprintf(stderr, "Can't open %s\n", src.c_str());
        return;
    }

    //set up error object
    cinfo.err = jpeg_std_error(&jerr);

    //initialize decompression object
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, in);

    //read JPG header
    jpeg_read_header(&cinfo, TRUE);

    //if not grascale, return
    if( cinfo.num_components != 1 )
    {
        cleanup();
        fprintf(stderr, "Not grayscale: %s, number of components: %d\n", src.c_str(), cinfo.num_components);
        return;
    }

    // Set values
    this->height = cinfo.image_height;
    this->width  = cinfo.image_width;
    this->pixels = width*height;
    this->valid  = true;

    //resize vector
    image.resize(width*height);

    //start decompressor
    jpeg_start_decompress(&cinfo);

    // Pointer to scanline
    buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr) &cinfo, JPOOL_IMAGE, width, 1);

    int lines_read=0;
    while( cinfo.output_scanline < cinfo.image_height )
    {
        jpeg_read_scanlines(&cinfo, buffer, 1);
        memcpy(&image[0]+lines_read*width, buffer[0], width);
        lines_read++;
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(in);
}

Grayscale :: Grayscale(const std::string& src, int width, int height) : image(width*height)
{
    FILE *in;
    if( (in = fopen(src.c_str(), "rb")) == NULL )
    {
        cleanup();
        fprintf(stderr, "Can't open %s\n", src.c_str());
    }

    this->height = height;
    this->width  = width;
    this->pixels = width*height;
    this->valid  = true;

    if( fread(&image[0], sizeof(uint8_t), pixels, in) != pixels )
    {
        fprintf(stderr,"Read JPG error\n");
        cleanup();
    }

    fclose(in);
}

Grayscale :: Grayscale(int width, int height) : image(width*height)
{
    this->height = height;
    this->width  = width;
    this->pixels = width*height;
    this->valid  = true;
}

void Grayscale :: cleanup()
{
    image.clear();
    this->valid = false;
}

Grayscale :: ~Grayscale() { cleanup(); }

int Grayscale :: get_pixels() { return pixels; };

int Grayscale :: get_height() { return height; }

int Grayscale :: get_width() { return width; }

inline void Grayscale :: get_pixel(uint8_t *color, int i, int j) { *color = image[i*width+j]; }

inline void Grayscale :: set_pixel(uint8_t color, int i, int j) { image[i*width+j] = color; }

inline uint8_t& Grayscale :: operator()(int i) { return image[i]; }

inline uint8_t& Grayscale :: operator()(int i, int j) { return image[i*width+j]; }

bool Grayscale :: is_valid() { return valid; }

int Grayscale :: write_raw(const std::string& des)
{
    if( !valid ) return -1;

    FILE* out;
    if( (out = fopen(des.c_str(), "wb")) == NULL )
        return -1;

    fwrite(&image[0], sizeof(uint8_t), pixels, out);
    fclose(out);
    return 1;
}

int Grayscale :: write_jpg(const std::string& des)
{
    if( !valid ) return -1;

    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    JSAMPROW row_p[1];
    FILE* out;

    //set up error handler
    cinfo.err = jpeg_std_error(&jerr);

    //initialize jpeg compression object
    jpeg_create_compress(&cinfo);

    //open file
    if( (out = fopen(des.c_str(), "wb")) == NULL )
    {

        fprintf(stderr, "Cannot open %s\n", des.c_str());
        return -1;
    }
    jpeg_stdio_dest(&cinfo, out);

    //set mandatory parameters
    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 1;
    cinfo.in_color_space = JCS_GRAYSCALE;

    //set default parameters
    jpeg_set_defaults(&cinfo);

    //set optional paramets
    jpeg_set_quality(&cinfo, 100, TRUE);

    //start compressor
    jpeg_start_compress(&cinfo, TRUE);

    while( cinfo.next_scanline < cinfo.image_height )
    {
        row_p[0] = &image[cinfo.next_scanline * width];
        jpeg_write_scanlines(&cinfo, row_p, 1);
    }

    //finish compress
    jpeg_finish_compress(&cinfo);
    fclose(out);
    jpeg_destroy_compress(&cinfo);

    return 1;
}

void Grayscale :: horizontal_flip()
{
    if( !valid ) return;
    for(int i=0 ; i<height ; i++)
        for(int j=0 ; j<width/2 ; j++)
            std::swap( image[i*width+j], image[i*width+(width-j-1)] );
}

void Grayscale :: vertical_flip()
{
    if( !valid ) return;
    for(int j=0 ; j<width ; j++)
        for(int i=0 ; i<height/2 ; i++)
            std::swap( image[i*width+j], image[(height-i-1)*width+j]);
}

void Grayscale :: rotate(uint8_t dir)
{
    if( !valid ) return;
    if(dir == 'L')
    {
        std::vector<uint8_t> rotated_image(pixels);
        for(int i=0 ; i<height ; i++)
            for(int j=0 ; j<width ; j++)
                rotated_image[(width-j-1)*height+i] = image[i*width+j];
        cleanup();
        std::swap(this->height, this->width);
        image.swap(rotated_image);
    }
    else if(dir == 'R')
    {
        std::vector<uint8_t> rotated_image(pixels);
        for(int i=0 ; i<height ; i++)
            for(int j=0 ; j<width ; j++)
                rotated_image[j*height+(height-i)] = image[i*width+j];
        cleanup();
        std::swap(this->height, this->width);
        image.swap(rotated_image);
    }
}

void Grayscale :: threshold(uint8_t t)
{
    if(!valid) return;
    for(int i=0 ; i<pixels ; i++)
        image[i] = image[i] < t ? 0 : 255;
}

void Grayscale :: gamma_correction(double gamma)
{
    if( !valid ) return;
    int max = -1;
    for(int i=0 ; i<pixels ; i++)
        if(max < image[i])
            max = image[i];

    double c = 255.0/pow(max,gamma);

    for(int i=0 ; i<pixels ; i++)
        image[i] = c * pow( image[i], gamma );
}

void Grayscale :: bit_plane_slice(uint8_t byte)
{
    if(!valid) return;
    for(int i=0 ; i<pixels ; i++)
        image[i] &= byte;
}

void Grayscale :: contrast_stretch(int r1, int s1, int r2, int s2)
{
    if(!valid) return;

    float stretch_factor = (float)(s2-s1)/(r2-r1);
    float left_factor    = (float)(s1)/(r1);
    float right_factor   = (float)(255-s2)/(255-r2);

    for(int i=0 ; i<pixels ; i++)
    {
        if(image[i] < r1)
            image[i] *= left_factor;
        else if(image[i] < r2)
            image[i] *= stretch_factor;
        else
            image[i] *= right_factor;
    }
}

void Grayscale :: negative()
{
    if(!valid) return;
    for(int i=0 ; i<pixels ; i++)
        image[i] = ~image[i];
}

void Grayscale :: heat_map()
{
}

void Grayscale :: hist_eq()
{
    if(!valid) return;

    int H[256] = {0};
    float pdf[256];
    float cdf[256];

    //histogram
    for(int i=0 ; i<pixels ; i++)
        H[image[i]]++;

    //pdf
    for(int i=0 ; i<256 ; i++)
        pdf[i] = (float)H[i]/pixels;

    //cdf
    cdf[0] = pdf[0];
    for(int i=1 ; i<256 ; i++)
        cdf[i] = cdf[i-1] + pdf[i];

    //max intensity
    int max=0;
    for(int i=0 ; i<pixels ; i++)
        if(image[i] > max)
            max = image[i];

    //map to new values
    for(int i=0 ; i<pixels ; i++)
        image[i] = (max * cdf[image[i]] + 0.5);
}

void Grayscale :: hist_spec(const std::string& src)
{
    if(!valid) return;

    Grayscale ref(src);
    int lookup[256];

    /* Current image:   X
     * Reference image: Z
     */

    int   Hx[256] = {0};             // histogram of X
    float Px[256];                   // pdf of X
    float Cx[256];                   // cdf of X

    int   Hz[256] = {0};             // histogram of Z
    float Pz[256];                   // pdf of Z
    float Cz[256];                   // cdf of Z

    /* Process current image */
    //Hx
    for(int i=0 ; i<pixels ; i++)
        Hx[image[i]]++;

    //Px
    for(int i=0 ; i<256 ; i++)
        Px[i] = (float)Hx[i]/pixels;

    //Cx
    Cx[0] = Px[0];
    for(int i=1 ; i<256 ; i++)
        Cx[i] = Cx[i-1] + Px[i];

    /* Process reference image */
    //Hz
    for(int i=0 ; i<ref.get_pixels() ; i++)
        Hz[ref(i)]++;

    //Pz
    for(int i=0 ; i<256 ; i++)
        Pz[i] = (float)Hz[i]/ref.get_pixels();

    //Cz
    Cz[0] = Pz[0];
    for(int i=1 ; i<256 ; i++)
        Cz[i] = Cz[i-1] + Pz[i];

    /* Calculate lookup
     * reference: http://fourier.eng.hmc.edu/e161/lectures/contrast_transform/node3.html
     */
    int j=0;
    for(int i=0 ; i<256 ; i++)
    {
        if(Cx[i] <= Cz[j])
            lookup[i] = j;
        else
        {
            while(Cx[i] > Cz[j])
                j++;

            if( Cz[j] - Cx[i] > Cx[i] - Cz[j-1] )
                lookup[i] = --j;
            else
                lookup[i] = j;
        }
    }

    /* Map to new values */
    for(int i=0 ; i<pixels ; i++)
        image[i] = lookup[image[i]];
}

void Grayscale :: pad(int padding, uint8_t value)
{
    if(!valid) return;

    int new_size = pixels + 2*padding*height + 2*padding*width + 4*padding*padding;
    std::vector<uint8_t> padded_image(new_size, value);

    for(int i=0, p=padding ; i<height ; i++, p++)
        for(int j=0, q=padding ; j<width ; j++, q++)
            padded_image[p*(width+2*padding)+q] = image[i*width+j];

    pixels  = new_size;
    width  += 2*padding;
    height += 2*padding;
    image.swap(padded_image);
}

void Grayscale :: pad_horizontal(int padding, uint8_t value)
{
	if(!valid) return;

    int new_size = pixels + 2*padding*height;
	int new_width = width + 2*padding;
    std::vector<uint8_t> padded_image(new_size, value);

    for(int i=0 ; i<height ; i++)
	{
		std::fill(padded_image.begin()+i*new_width, padded_image.begin()+i*new_width+padding, value);
        std::copy(image.begin()+i*width, image.begin()+(i+1)*width, padded_image.begin()+i*new_width+padding);
		std::fill(padded_image.begin()+i*new_width+padding+width, padded_image.begin()+(i+1)*new_width, value);
	}

    pixels = new_size;
    width  = new_width;
    image.swap(padded_image);
}

void Grayscale :: pad_vertical(int padding, uint8_t value)
{
	if(!valid) return;

	int new_size = pixels + 2*padding*width;
	int new_height = height + 2*padding;
    std::vector<uint8_t> padded_image(new_size, value);

	std::fill(padded_image.begin(), padded_image.begin()+padding*width, value);
    std::copy(image.begin(), image.end(), padded_image.begin()+padding*width);
	std::fill(padded_image.begin()+padding*width+pixels, padded_image.end(), value);

    pixels = new_size;
    height = new_height;
    image.swap(padded_image);
}

void Grayscale :: conv(float* kernel, int n)
{
    if(!valid) return;

    float sum;
    Grayscale padded_image(*this);
    padded_image.pad(n/2, 0);
    
    int pheight = padded_image.get_height();
    int pwidth  = padded_image.get_width();

    for(int i=0,ii=0 ; i<=pheight-n ; i++,ii++)
    {
        for(int j=0,jj=0 ; j<=pwidth-n ; j++,jj++)
        {
            sum = 0;
            for(int x=0 ; x<n ; x++)
                for(int y=0 ; y<n ; y++)
                    sum += padded_image(i+x,j+y) * kernel[x*n+y];
            set_pixel(abs(sum), ii, jj);
        }
    }
}

void Grayscale :: mean_filter(int n)
{
    if(!valid) return;

    float sum;
    float *kernel = new float[n*n];
    for(int i=0 ; i<n*n ; i++)
        kernel[i] = 1.0f/(n*n);

    Grayscale padded_image(*this);
    padded_image.pad(n/2, 0);
    
    for(int i=0 ; i<height ; i++)
    {
        for(int j=0 ; j<width ; j++)
        {
            sum = 0;
            for(int x=0 ; x<n ; x++)
                for(int y=0 ; y<n ; y++)
                    sum += padded_image(i+x,j+y) * kernel[x*n+y];

            set_pixel(sum, i, j);
        }
    }
	delete[] kernel;
}

void Grayscale :: median_filter(int n)
{   
    if(!valid) return;

    int *values = new int[n*n];
    Grayscale padded_image(*this);
    padded_image.pad(n/2, 0);
    
    for(int i=0 ; i<height ; i++)
    {
        for(int j=0 ; j<width ; j++)
        {
            int k=0;
            for(int x=0 ; x<n ; x++)
                for(int y=0 ; y<n ; y++)
                    values[k++] = padded_image(i+x,j+y);

            std::nth_element(values, values+(n*n)/2, values+n*n);
            set_pixel(values[(n*n)/2], i, j);
        }
    }
	delete[] values;
}

void Grayscale :: unsharp_filter(int kernel_size, float weight)
{
    if(!valid) return;

    Grayscale mask(*this);
    mask.mean_filter(kernel_size);

    for(int i=0 ; i<pixels ; i++)
        mask(i) = image[i] - mask(i);

    for(int i=0 ; i<pixels ; i++)
        image[i] = image[i] + weight * mask(i);
}

void Grayscale :: max_neighbour(int n)
{
    if(!valid) return;

    Grayscale padded_image(*this);
    padded_image.pad(n/2, 0);
    
    for(int i=0 ; i<height ; i++)
    {
        for(int j=0 ; j<width ; j++)
        {
            int max = -1;
            for(int x=0 ; x<n ; x++)
                for(int y=0 ; y<n ; y++)
                    if(max < padded_image(i+x,j+y))
                        max = padded_image(i+x,j+y);

            set_pixel(max, i, j);
        }
    }
}

void Grayscale :: min_neighbour(int n)
{
    if(!valid) return;

    Grayscale padded_image(*this);
    padded_image.pad(n/2, 0);
    
    for(int i=0 ; i<height ; i++)
    {
        for(int j=0 ; j<width ; j++)
        {
            int min = 256;
            for(int x=0 ; x<n ; x++)
                for(int y=0 ; y<n ; y++)
                    if(min > padded_image(i+x,j+y))
                        min = padded_image(i+x,j+y);

            set_pixel(min, i, j);
        }
    }
}

void Grayscale :: save_hist(const std::string& des)
{
    if(!valid) return;

    Grayscale hist(256,256);
    int H[256] = {0};

    for(int i=0 ; i<pixels ; i++)
        H[image[i]]++;

    int maxcount=0;
    for(int i=0 ; i<256 ; i++)
        if(maxcount < H[i]) maxcount = H[i];

    float c = (float)maxcount/255;
    
    for(int j=0 ; j<256 ; j++)
    {
        int barheight = H[j]/c;
        int row=255;
        while(barheight--)
            hist.set_pixel(255, row--, j);
    }
    hist.write_jpg(des);
}

void Grayscale :: edge_sobel()
{
    float sobel_y[9] =
    {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };
    float sobel_x[9] =
    {
        -1,-2,-1,
         0, 0, 0,
         1, 2, 1
    };
    Grayscale Dx(*this);
    Grayscale Dy(*this);
    Dx.conv(sobel_y,3);
    Dy.conv(sobel_x,3);
    for(int i=0 ; i<pixels ; i++)
        image[i] = pow(Dx(i)*Dx(i)+Dy(i)*Dy(i),0.5);
}

void Grayscale :: edge_prewitt()
{
    float prewitt_y[9] =
    {
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1
    };
    float prewitt_x[9] =
    {
        -1,-1,-1,
         0, 0, 0,
         1, 1, 1
    };
    Grayscale Dx(*this);
    Grayscale Dy(*this);
    Dx.conv(prewitt_y,3);
    Dy.conv(prewitt_x,3);
    for(int i=0 ; i<pixels ; i++)
        image[i] = pow(Dx(i)*Dx(i)+Dy(i)*Dy(i),0.5);
}

void Grayscale :: erode(bool *se, int s_width, int s_height)
{
	if(!valid) return;
	
    Grayscale padded_image(*this);
    padded_image.pad_horizontal(s_width/2, 0);
	padded_image.pad_vertical(s_height/2, 0);
    
    int pheight = padded_image.get_height();
    int pwidth  = padded_image.get_width();

    for(int i=0,ii=0 ; i<=pheight-s_height ; i++,ii++)
    {
        for(int j=0,jj=0 ; j<=pwidth-s_width ; j++,jj++)
        {
            set_pixel(255, ii, jj);
			for(int x=0 ; x<s_height ; x++)
                for(int y=0 ; y<s_width ; y++)
					if(se[x*s_width+y] == 1 && padded_image(i+x,j+y) != 255)
					{ set_pixel(0, ii, jj); break; }
        }
    }
}

void Grayscale :: dilate(bool *se, int s_width, int s_height)
{
	if(!valid) return;
	
    Grayscale padded_image(*this);
    padded_image.pad_horizontal(s_width/2, 0);
	padded_image.pad_vertical(s_height/2, 0);
    
    int pheight = padded_image.get_height();
    int pwidth  = padded_image.get_width();

    for(int i=0,ii=0 ; i<=pheight-s_height ; i++,ii++)
    {
        for(int j=0,jj=0 ; j<=pwidth-s_width ; j++,jj++)
        {
            set_pixel(0, ii, jj);
			for(int x=0 ; x<s_height ; x++)
                for(int y=0 ; y<s_width ; y++)
					if(se[x*s_width+y] == 1 && padded_image(i+x,j+y) == 255)
					{ set_pixel(255, ii, jj); break; }
        }
    }
}

void Grayscale :: open(bool *se, int s_width, int s_height)
{
	if(!valid) return;
	erode(se, s_width, s_height);
	dilate(se, s_width, s_height);
}

void Grayscale :: close(bool *se, int s_width, int s_height)
{
	if(!valid) return;
	dilate(se, s_width, s_height);
	erode(se, s_width, s_height);
}

void Grayscale :: boundary(bool *se, int s_width, int s_height)
{
	if(!valid) return;
	Grayscale eroded(*this);
	eroded.erode(se, s_width, s_height);
	for(int i=0 ; i<pixels ; i++)
		image[i] = image[i] - eroded(i);
}

void Grayscale :: draw_line(double m, double c)
{
    //naive line drawing
	for(int x=0 ; x<width ; x++)
	{
		int y = m * x + c + 0.5;
		int index = y*width+x;
		if(index>=0 && index<pixels && y>=0)
			image[index] = 255;
	}
}

void Grayscale :: hough(int threshold)
{
	int dmax = std::ceil(std::pow(width*width+height*height,0.5));
	int accheight = 2*dmax+1;
	int accwidth = 180;
	std::vector<int> theta(accwidth);
	std::vector<int> rho(accheight);
	std::vector<int> acc(accwidth*accheight, 0);
	std::vector<double> tsin(accwidth);
	std::vector<double> tcos(accwidth);
	double pi = acos(-1);

	// Precompute
	for(int i=0, t=-90 ; i<accwidth ; i++, t++)
		theta[i] = t;
	
	for(int i=0, r=-dmax ; i<accheight ; i++, r++)
		rho[i] = r;

	// Make tables of sin and cos
	for(int i=0 ; i<180 ; i++)
	{
		tcos[i] = cos(theta[i] * pi / 180.0);
		tsin[i] = sin(theta[i] * pi / 180.0);
	}
	
	// Construct accumulator
	for(int x=0 ; x<width ; x++)
		for(int y=0 ; y<height ; y++)
			if(image[y*width+x] != 0)
			{
				for(int itheta = 0 ; itheta < accwidth ; itheta++)
				{
					//Distance from origin given t
					int dist = x*tcos[itheta] + y*tsin[itheta] + 0.5;

					// Find closest rho value to dist
					int irho;
					int lo=0, hi=accheight-1;
					while(lo <= hi)
					{
						irho = (lo+hi)/2;
						if(rho[irho] == dist) break;
						else if(rho[irho] < dist) lo = irho+1;
						else hi = irho-1;
					}

					//Increment accumulator
					acc[irho*accwidth+itheta] = acc[irho*accwidth+itheta] + 1;
				}
			}

	// Draw lines
	for(int irho=0 ; irho<accheight ; irho++)
		for(int itheta=0 ; itheta<accwidth ; itheta++)
			if(acc[irho*accwidth+itheta] >= threshold)
				draw_line(-tcos[itheta]/tsin[itheta] , rho[irho]/tsin[itheta]);

	//Save parameter space as image
	Grayscale para(accwidth, accheight);
	for(int i=0 ; i<acc.size() ; i++)
		para(i) = acc[i];
	para.write_jpg("test/para.jpg");

	std::cerr << "Hough transform done." << std::endl;
	std::cerr << "Parameter space image: para.jpg" << std::endl;
}