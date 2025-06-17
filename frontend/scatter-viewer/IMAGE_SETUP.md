# Image Setup for ScatterLabel Viewer

The ScatterLabel viewer displays images corresponding to selected annotations. To set up image serving:

## Option 1: Local Static File Server (Development)

1. Create an `images` directory in the public folder:
   ```bash
   mkdir frontend/scatter-viewer/public/images
   ```

2. Copy your images to this directory. Images should be named using their `image_id` from the CSV:
   ```
   public/images/
   ├── 120d-2025-01-10T21:07:46.372989359Z-185.jpg
   ├── 120d-2025-05-12T12:17:48.037475075Z-138.jpg
   └── ...
   ```

3. The default `imageBasePath` in the app is set to `/images` which will serve from the public directory.

## Option 2: External Image Server

1. If your images are hosted on a separate server, update the `imageBasePath` in `App.jsx`:
   ```jsx
   <ImageGrid 
     selectedData={selectedData} 
     imageBasePath="http://your-image-server.com/images"
   />
   ```

2. Ensure CORS is properly configured on your image server to allow the React app to fetch images.

## Option 3: Local File System (Using a Simple HTTP Server)

1. If your images are stored elsewhere on your file system, you can run a simple HTTP server:
   ```bash
   # Using Python 3
   cd /path/to/your/images
   python -m http.server 8080
   ```

2. Update the `imageBasePath` in `App.jsx`:
   ```jsx
   <ImageGrid 
     selectedData={selectedData} 
     imageBasePath="http://localhost:8080"
   />
   ```

## Image Format

- Images should be in JPEG format (`.jpg` extension)
- The ImageGrid component expects images to be named as `{image_id}.jpg`
- If your images have different extensions or naming conventions, modify the image loading logic in `ImageGrid.jsx`:
  ```jsx
  img.src = `${imageBasePath}/${imageId}.jpg`; // Change extension here
  ```

## Troubleshooting

- If images aren't loading, check the browser console for 404 errors
- Verify the image paths match the `image_id` values in your CSV data
- Ensure your image server is running and accessible
- Check for CORS issues if using an external server