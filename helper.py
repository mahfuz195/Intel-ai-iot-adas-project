import math
import numpy as np
import cv2

def to_hsv(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2HSV)

def to_hls(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2HLS)

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def isolate_yellow_hls(img):
    
    low  = np.array([20,  100, 100],dtype=np.uint8)
    high = np.array([30, 255, 255],dtype=np.uint8)
    
    mask = cv2.inRange(img, low, high)

    return mask
def isolate_white_hls(img):
    
    low  = np.array([  20,  0, 180],dtype=np.uint8)
    high = np.array([255,  80, 255],dtype=np.uint8)
    
    mask = cv2.inRange(img, low, high)
    
    return mask
    

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def create_vertices(img):
    
    ysize, xsize = img.shape[0], img.shape[1]
    bottom_ignore = ysize//6
    ybuffer = ysize//30
    xbuffer_top = xsize//50
    xbuffer_bot = xbuffer_top*2
    side_search_buffer = ybuffer//2
    """
    # Let's find the last white pixel's index in the center column.
    # This will give us an idea of where our region should be
    # We ignore a certain portion of the bottom of the screen so we get a better region top
    #   - This is partly because car hoods can obsure the region
    center_white = img[:ysize-bottom_ignore, xsize//2] == 255
    indices = np.arange(0, center_white.shape[0])
    indices[~center_white] = 0
    last_white_ind = np.amax(indices)
    
    # If our first white pixel is too close to the bottom of the screen, default back to the screen center
    # region_top_y = (last_white_ind if last_white_ind < 4*ysize//5 else ysize//2) + ybuffer
    region_top_y = min(last_white_ind + ybuffer, ysize-1)
    
    # Now we need to find the x-indices for the top segment of our region
    # To do this we will look left and right from our center point until we find white
    y_slice_top = max(region_top_y - side_search_buffer, 0)
    y_slice_bot = min(region_top_y + side_search_buffer, ysize-1)
    region_top_white = np.copy(img[y_slice_top:y_slice_bot, :]) == 255
    
    indices = np.zeros_like(region_top_white, dtype='int32')
    indices[:, :] = np.arange(0, xsize)
    indices[~region_top_white] = 0
    
    # Separate into right and left sides we can grab our indices easier:
    # Right side min and left side max
    right_side = np.copy(indices)
    right_side[right_side < xsize//2] = xsize*2  # Large number because we will take min
    left_side = np.copy(indices)
    left_side[left_side > xsize//2] = 0
    
    region_top_x_left = max(np.amax(left_side) - xbuffer_top, 0)
    region_top_x_right = min(np.amin(right_side) + xbuffer_top, xsize)
    
    # Now we do the same thing for the bottom
    # Look left and right from the center until we hit white
    indices = np.arange(0, xsize)
    region_bot_white = img[ysize-bottom_ignore, :] == 255
    indices[~region_bot_white] = 0
    
    # Separate into right and left sides we can grab our indices easier:
    # Right side min and left side max
    right_side = np.copy(indices)
    right_side[right_side < xsize//2] = xsize*2  # Large number because we will take min
    left_side = np.copy(indices)
    left_side[left_side > xsize//2] = 0
    
    region_bot_x_left = max(np.amax(left_side) - xbuffer_bot, 0)
    region_bot_x_right = min(np.amin(right_side) + xbuffer_bot, xsize)
    
    # Because of our bottom_ignore, we need to extrapolate these bottom x coords to bot of screen
    left_slope = ((ysize-bottom_ignore) - region_top_y)/(region_bot_x_left - region_top_x_left)
    right_slope = ((ysize-bottom_ignore) - region_top_y)/(region_bot_x_right - region_top_x_right)
    # Let's check these slopes we don't divide by 0 or inf
    if abs(left_slope < .001):
        left_slope = .001 if left_slope > 0 else -.001
    if abs(right_slope < .001):
        right_slope = .001 if right_slope > 0 else -.001
    if abs(left_slope) > 1000:
        left_slope = 1000 if left_slope > 0 else -1000
    if abs(right_slope) > 1000:
        right_slope = 1000 if right_slope > 0 else -1000
    
    # b=y-mx
    left_b = region_top_y - left_slope*region_top_x_left
    right_b = region_top_y - right_slope*region_top_x_right
    # x=(y-b)/m
    region_bot_x_left = max(int((ysize-1-left_b)/left_slope), 0)
    region_bot_x_right = min(int((ysize-1-right_b)/right_slope), xsize-1)
    
    region_bot_x_right = xsize-1
    verts = [
        (region_bot_x_left, ysize),
        (region_top_x_left, region_top_y),
        (region_top_x_right, region_top_y),
        (region_bot_x_right, ysize)
    ]
    """
    verts = [
        (100, ysize),
        (650, 460),
        (870, 460),
        (1278, 640)
    ]
        
    
    
    #line_info = [(left_b,left_slope),(right_b,right_slope)]
    
    return np.array([verts], dtype=np.int32)

def region_of_interest(img):
    #defining a blank mask to start with
    mask = np.zeros_like(img)
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    verts  = create_vertices(img)
    cv2.fillPoly(mask, verts, ignore_mask_color)
    
    print (verts)
    #Let's return an image of the regioned area in lines
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #cv2.fillPoly(line_img, [verts], 255)
    cv2.polylines(line_img, verts, isClosed=True, color=[0, 255, 0], thickness=4)
    
    v = np.concatenate(verts).ravel().tolist()
    cv2.circle(line_img,(v[0],v[1]), 20, (0,255,0),-1)
    cv2.circle(line_img,(v[2],v[3]), 20, (0,255,0),-1)
    cv2.circle(line_img,(v[4],v[5]), 20, (0,255,0),-1)
    cv2.circle(line_img,(v[6],v[7]), 20, (0,255,0),-1)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    
    cv2.circle(masked_image,(v[0],v[1]), 20, (0,255,0),-1)
    cv2.circle(masked_image,(v[2],v[3]), 20, (0,255,0),-1)
    cv2.circle(masked_image,(v[4],v[5]), 20, (0,255,0),-1)
    cv2.circle(masked_image,(v[6],v[7]), 20, (0,255,0),-1)
    
    return masked_image, line_img

def draw_lines(img, lines, color=[255, 0, 0], thickness=8):
    if lines is None: return lines
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    avg_lines = average_lines(lines, img)
    
    #print (avg_lines)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
#     draw_lines(line_img, lines)
    #draw_lines(line_img, avg_lines, color=[138,43,226])
    #print (avg_lines[0][0][0])
    #print ('==')
    #print (avg_lines[1])
    
    
    #cv2.fillPoly(line_img,avg_lines,255)
    
    draw_lines(line_img, avg_lines, color=[0,255,0])
    #print (len(avg_lines))
    if(avg_lines is not None):
        if(len(avg_lines)>1):
            avg_lines = np.concatenate(avg_lines).ravel().tolist()
    
    #print (len(avg_lines))
    verts_ = np.array([[ 0,0],[ 0,0],[ 0, 0], [0,0]])
    if(avg_lines is not None):
        if(len(avg_lines)>6):
            verts_ = np.array([[avg_lines[0],avg_lines[1]],
                       [avg_lines[4],avg_lines[5]],
                       [avg_lines[6],avg_lines[7]],
                       [avg_lines[2],avg_lines[3]]])
        
            poly_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            cv2.fillPoly(poly_img, pts = [verts_], color = (0,255,0))
            line_img = weighted_img(line_img,poly_img,a=1.0, b=.2, l=0.)
        
        
    return line_img, verts_


prev_left = []
prev_right = []

def average_lines(lines, img):
    '''
    img should be a regioned canny output
    '''
    if lines is None: return lines
    global prev_left, prev_right
    
    positive_slopes = []
    positive_xs = []
    positive_ys = []
    negative_slopes = []
    negative_xs = []
    negative_ys = []
    
    min_slope = .3
    max_slope = 1000
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2-y1)/(x2-x1)
            
            if abs(slope) < min_slope or abs(slope) > max_slope: continue  # Filter our slopes
                
            # We only need one point sample and the slope to determine the line equation
            positive_slopes.append(slope) if slope > 0 else negative_slopes.append(slope)
            positive_xs.append(x1) if slope > 0 else negative_xs.append(x1)
            positive_ys.append(y1) if slope > 0 else negative_ys.append(y1)
    
    # We need to calculate our region_top_y from the canny image so we know where to extend our lines to
    ysize, xsize = img.shape[0], img.shape[1]
    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    white = img == 255
    YY[~white] = ysize*2  # Large number because we will take the min
    
    region_top_y = np.amin(YY)
    
    new_lines = []
    if len(positive_slopes) > 0:
        m = np.mean(positive_slopes)
        avg_x = np.mean(positive_xs)
        avg_y = np.mean(positive_ys)
        
        b = avg_y - m*avg_x
        
        # We have m and b, so with a y we can get x = (y-b)/m
        x1 = int((region_top_y - b)/m)
        x2 = int((ysize - b)/m)
        prev_left = [(x1, region_top_y, x2, ysize)]
        new_lines.append([(x1, region_top_y, x2, ysize)])
    else:
        if(len(prev_left)>0):new_lines.append(prev_left)
        
    
    if len(negative_slopes) > 0:
        m = np.mean(negative_slopes)
        avg_x = np.mean(negative_xs)
        avg_y = np.mean(negative_ys)
        
        b = avg_y - m*avg_x
        
        # We have m and b, so with a y we can get x = (y-b)/m
        x1 = int((region_top_y - b)/m)
        x2 = int((ysize - b)/m)
        
        prev_right = [(x1, region_top_y, x2, ysize)]
        new_lines.append([(x1, region_top_y, x2, ysize)])
    else:
        if(len(prev_right)>0):new_lines.append(prev_right)
    
    
    return np.array(new_lines)

def weighted_img(initial_img, img, a=0.8, b=1., l=0.):
    return cv2.addWeighted(initial_img, a, img, b, l)

def save_img(img, name):
    mpimg.imsave('./images/output/{0}'.format(name if '.' in name else '{0}.png'.format(name)), img)