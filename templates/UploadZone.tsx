import { useState, useRef, DragEvent, ChangeEvent } from 'react';
import { Upload, Image, AlertCircle } from 'lucide-react';

interface UploadZoneProps {
  onFileSelect: (file: File) => void;
  isLoading: boolean;
}

const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];

export const UploadZone = ({ onFileSelect, isLoading }: UploadZoneProps) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const validateFile = (file: File): string | null => {
    if (!ALLOWED_TYPES.includes(file.type)) {
      return 'Invalid file type. Please upload JPG, PNG, WebP, or GIF.';
    }
    if (file.size > MAX_FILE_SIZE) {
      return `File too large. Maximum size is ${MAX_FILE_SIZE / 1024 / 1024}MB.`;
    }
    return null;
  };

  const handleFile = (file: File) => {
    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      return;
    }

    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(file);
    onFileSelect(file);
  };

  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  };

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const handleClick = () => {
    if (!isLoading) inputRef.current?.click();
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div
        onClick={handleClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`upload-zone p-8 md:p-12 cursor-pointer relative overflow-hidden ${
          isDragOver ? 'drag-over' : ''
        } ${isLoading ? 'pointer-events-none opacity-60' : ''}`}
      >
        {/* Animated glow effect */}
        <div className="absolute inset-0 bg-gradient-to-r from-primary/0 via-primary/5 to-primary/0 opacity-0 group-hover:opacity-100 transition-opacity" />
        
        <input
          ref={inputRef}
          type="file"
          accept={ALLOWED_TYPES.join(',')}
          onChange={handleInputChange}
          className="hidden"
        />

        <div className="flex flex-col items-center gap-4 text-center relative z-10">
          {preview && !isLoading ? (
            <div className="relative">
              <img
                src={preview}
                alt="Preview"
                className="max-h-48 rounded-lg border border-border/50 shadow-lg"
              />
              <div className="absolute inset-0 bg-background/60 backdrop-blur-sm flex items-center justify-center rounded-lg opacity-0 hover:opacity-100 transition-opacity">
                <span className="text-sm font-medium">Click to change</span>
              </div>
            </div>
          ) : (
            <>
              <div className="w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center float-animation">
                {isLoading ? (
                  <div className="spinner" />
                ) : (
                  <Upload className="w-10 h-10 text-primary" />
                )}
              </div>
              <div>
                <h3 className="text-xl font-display font-semibold mb-2 text-foreground">
                  {isLoading ? 'Analyzing Image...' : 'Upload Medical Image'}
                </h3>
                <p className="text-muted-foreground text-sm max-w-sm">
                  {isLoading
                    ? 'Please wait while we extract your health metrics'
                    : 'Drag and drop your blood pressure monitor image or click to browse'}
                </p>
              </div>
              <div className="flex items-center gap-4 text-xs text-muted-foreground">
                <span className="flex items-center gap-1">
                  <Image className="w-3 h-3" />
                  JPG, PNG, WebP, GIF
                </span>
                <span>Max 10MB</span>
              </div>
            </>
          )}
        </div>
      </div>

      {error && (
        <div className="mt-4 p-4 rounded-lg bg-destructive/10 border border-destructive/30 flex items-start gap-3 animate-fade-in">
          <AlertCircle className="w-5 h-5 text-destructive shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-medium text-destructive">{error}</p>
            <p className="text-xs text-muted-foreground mt-1">
              Please select a valid image file and try again.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};