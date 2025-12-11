import { Activity, Zap } from 'lucide-react';

export const Header = () => {
  return (
    <header className="relative py-8 md:py-12">
      {/* Background glow */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[300px] bg-primary/10 blur-[100px] rounded-full" />
      </div>

      <div className="relative text-center">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 border border-primary/20 mb-6 animate-fade-in">
          <Zap className="w-4 h-4 text-primary" />
          <span className="text-sm font-medium text-primary">AI-Powered OCR</span>
        </div>

        <div className="flex items-center justify-center gap-3 mb-4">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center shadow-lg">
            <Activity className="w-6 h-6 text-primary-foreground" />
          </div>
          <h1 className="text-3xl md:text-5xl font-display font-bold">
            <span className="text-gradient">BioNexus</span>
            <span className="text-foreground"> Reader</span>
          </h1>
        </div>

        <p className="text-muted-foreground max-w-lg mx-auto text-sm md:text-base">
          Advanced medical OCR technology to extract and analyze health metrics
          from your blood pressure monitor readings
        </p>
      </div>
    </header>
  );
};
