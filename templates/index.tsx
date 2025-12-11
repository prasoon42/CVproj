import { useState } from 'react';
import { Header } from '@/components/Header';
import { UploadZone } from '@/components/UploadZone';
import { ResultsDisplay } from '@/components/ResultsDisplay';
import { AlertCircle, RefreshCw } from 'lucide-react';

interface Reading {
  id: string;
  timestamp: Date;
  sys: number;
  dia: number;
  pulse: number;
}

interface AnalysisResult {
  sys: number | null;
  dia: number | null;
  pulse: number | null;
}

const Index = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [history, setHistory] = useState<Reading[]>([]);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = async (file: File) => {
    setIsLoading(true);
    setError(null);

    try {
      // Simulate API call - replace with actual /analyze endpoint
      await new Promise((resolve) => setTimeout(resolve, 2000));

      // Mock response - in production, this would come from the backend OCR
      const mockResult: AnalysisResult = {
        sys: Math.floor(Math.random() * 40) + 110,
        dia: Math.floor(Math.random() * 30) + 65,
        pulse: Math.floor(Math.random() * 40) + 60,
      };

      setResult(mockResult);

      // Add to history if values detected
      if (mockResult.sys && mockResult.dia && mockResult.pulse) {
        const newReading: Reading = {
          id: Date.now().toString(),
          timestamp: new Date(),
          sys: mockResult.sys,
          dia: mockResult.dia,
          pulse: mockResult.pulse,
        };
        setHistory((prev) => [newReading, ...prev].slice(0, 10));
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Failed to analyze image. Please try again.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen">
      {/* Background effects */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-1/4 -left-1/4 w-[500px] h-[500px] bg-primary/5 rounded-full blur-[120px]" />
        <div className="absolute bottom-1/4 -right-1/4 w-[500px] h-[500px] bg-accent/5 rounded-full blur-[120px]" />
      </div>

      <div className="relative container mx-auto px-4 pb-12">
        <Header />

        <main className="max-w-5xl mx-auto">
          {/* Upload Section */}
          <section className="mb-12">
            <UploadZone onFileSelect={handleFileSelect} isLoading={isLoading} />
          </section>

          {/* Error Display */}
          {error && (
            <div className="mb-8 p-4 rounded-xl bg-destructive/10 border border-destructive/30 flex items-start gap-3 animate-fade-in max-w-2xl mx-auto">
              <AlertCircle className="w-5 h-5 text-destructive shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="text-sm font-medium text-destructive">{error}</p>
                <p className="text-xs text-muted-foreground mt-1">
                  Make sure the image clearly shows the blood pressure monitor display.
                </p>
              </div>
              <button
                onClick={handleReset}
                className="p-2 rounded-lg hover:bg-destructive/20 transition-colors"
              >
                <RefreshCw className="w-4 h-4 text-destructive" />
              </button>
            </div>
          )}

          {/* Results Section */}
          {result && !isLoading && (
            <section>
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-display font-semibold">
                  Analysis Results
                </h2>
                <button
                  onClick={handleReset}
                  className="flex items-center gap-2 px-4 py-2 rounded-lg bg-secondary hover:bg-secondary/80 transition-colors text-sm font-medium"
                >
                  <RefreshCw className="w-4 h-4" />
                  New Scan
                </button>
              </div>
              <ResultsDisplay result={result} history={history} />
            </section>
          )}

          {/* Empty State */}
          {!result && !isLoading && !error && (
            <div className="text-center py-12 animate-fade-in">
              <div className="inline-flex items-center gap-2 text-muted-foreground text-sm">
                <div className="w-2 h-2 rounded-full bg-primary pulse-animation" />
                <span>Ready to analyze your health metrics</span>
              </div>
            </div>
          )}
        </main>

        {/* Footer */}
        <footer className="mt-16 text-center text-xs text-muted-foreground">
          <p>
            BioNexus Reader is for informational purposes only.
            <br />
            Always consult a healthcare professional for medical advice.
          </p>
        </footer>
      </div>
    </div>
  );
};

export default Index;