import React, { useState, useEffect } from 'react';

const ImageGrid = ({ selectedData }) => {
  const [loadedImages, setLoadedImages] = useState({});
  const [imageErrors, setImageErrors] = useState({});
  const [imageDimensions, setImageDimensions] = useState({});

  // For CMR crops, each annotation has its own cropped image
  const prepareCroppedImages = (points) => {
    // Each point represents a single cropped annotation
    return points.filter(p => p.cropped_image_path);
  };

  // Preload images when selection changes
  useEffect(() => {
    // For CMR data, we now have cropped images
    const imagesToLoad = selectedData.filter(p => p.cropped_image_path);
    
    imagesToLoad.forEach(point => {
      const imageKey = point.cropped_image_path;
      if (!loadedImages[imageKey] && !imageErrors[imageKey]) {
        const img = new Image();
        img.onload = () => {
          setLoadedImages(prev => ({ ...prev, [imageKey]: img.src }));
          setImageDimensions(prev => ({ 
            ...prev, 
            [imageKey]: { width: img.naturalWidth, height: img.naturalHeight } 
          }));
        };
        img.onerror = () => {
          setImageErrors(prev => ({ ...prev, [imageKey]: true }));
        };
        // Adjust path - remove the 'cmr_crops_sample/' or 'cmr_crops_limited/' prefix
        const filename = point.cropped_image_path.split('/').pop();
        img.src = `/cmr_crops/${filename}`;
      }
    });
  }, [selectedData, loadedImages, imageErrors]);

  const croppedImages = prepareCroppedImages(selectedData);

  if (selectedData.length === 0) {
    return (
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        color: '#666',
        fontSize: '18px'
      }}>
        Select points in the scatter plot to view images
      </div>
    );
  }

  return (
    <div style={{
      padding: '20px',
      height: '100%',
      overflowY: 'auto',
      backgroundColor: '#f5f5f5'
    }}>
      <div style={{
        marginBottom: '20px',
        fontSize: '16px',
        fontWeight: 'bold',
        color: '#333'
      }}>
        {croppedImages.length} cropped annotations selected
      </div>
      
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
        gap: '15px'
      }}>
        {croppedImages.map((annotation) => {
          const imageKey = annotation.cropped_image_path;
          const filename = imageKey.split('/').pop();
          
          return (
            <div
              key={`${annotation.annotation_id}_${annotation.index}`}
              style={{
                backgroundColor: 'white',
                borderRadius: '8px',
                boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                overflow: 'hidden'
              }}
            >
              <div style={{ position: 'relative' }}>
                {loadedImages[imageKey] ? (
                  <img
                    src={loadedImages[imageKey]}
                    alt={`${annotation.class_name} crop`}
                    style={{
                      width: '100%',
                      height: 'auto',
                      display: 'block'
                    }}
                  />
                ) : imageErrors[imageKey] ? (
                  <div style={{
                    height: '150px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    backgroundColor: '#f0f0f0',
                    color: '#666'
                  }}>
                    <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '20px', marginBottom: '8px' }}>⚠️</div>
                      <div style={{ fontSize: '12px' }}>Image not found</div>
                    </div>
                  </div>
                ) : (
                  <div style={{
                    height: '150px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    backgroundColor: '#f0f0f0'
                  }}>
                    <div style={{ fontSize: '12px', color: '#666' }}>Loading...</div>
                  </div>
                )}
              </div>
              
              <div style={{ padding: '10px' }}>
                <div style={{
                  fontSize: '14px',
                  fontWeight: 'bold',
                  marginBottom: '4px'
                }}>
                  {annotation.class_name}
                </div>
                <div style={{
                  fontSize: '11px',
                  color: '#666'
                }}>
                  From: {annotation.image_id}
                </div>
                <div style={{
                  fontSize: '11px',
                  color: '#888',
                  marginTop: '2px'
                }}>
                  Bbox: [{Math.round(annotation.x_min)}, {Math.round(annotation.y_min)}, 
                        {Math.round(annotation.x_max)}, {Math.round(annotation.y_max)}]
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default ImageGrid;