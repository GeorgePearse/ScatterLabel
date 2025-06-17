import { useState, useCallback } from 'react';

const useImageLoader = (imageBasePath = '/images') => {
  const [loadedImages, setLoadedImages] = useState({});
  const [imageErrors, setImageErrors] = useState({});
  const [imageDimensions, setImageDimensions] = useState({});
  const [loadingImages, setLoadingImages] = useState({});

  const loadImage = useCallback((imageId) => {
    if (loadedImages[imageId] || imageErrors[imageId] || loadingImages[imageId]) {
      return;
    }

    setLoadingImages(prev => ({ ...prev, [imageId]: true }));

    const img = new Image();
    
    img.onload = () => {
      setLoadedImages(prev => ({ ...prev, [imageId]: img.src }));
      setImageDimensions(prev => ({ 
        ...prev, 
        [imageId]: { width: img.naturalWidth, height: img.naturalHeight } 
      }));
      setLoadingImages(prev => {
        const newState = { ...prev };
        delete newState[imageId];
        return newState;
      });
    };
    
    img.onerror = () => {
      setImageErrors(prev => ({ ...prev, [imageId]: true }));
      setLoadingImages(prev => {
        const newState = { ...prev };
        delete newState[imageId];
        return newState;
      });
    };
    
    img.src = `${imageBasePath}/${imageId}.jpg`;
  }, [imageBasePath, loadedImages, imageErrors, loadingImages]);

  const loadImages = useCallback((imageIds) => {
    imageIds.forEach(loadImage);
  }, [loadImage]);

  const clearCache = useCallback(() => {
    setLoadedImages({});
    setImageErrors({});
    setImageDimensions({});
    setLoadingImages({});
  }, []);

  return {
    loadedImages,
    imageErrors,
    imageDimensions,
    loadingImages,
    loadImages,
    clearCache
  };
};

export default useImageLoader;