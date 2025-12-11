import { Activity, Heart, Brain, TrendingUp } from 'lucide-react';

export type BPCategory = 'normal' | 'elevated' | 'high1' | 'high2' | 'crisis';

interface HealthInsightsProps {
  category: BPCategory;
  pulseAnalysis: string;
  recommendations: string[];
}

const categoryConfig = {
  normal: {
    label: 'Normal',
    color: 'bg-success',
    position: '12.5%',
    description: 'Your blood pressure is within the healthy range.',
  },
  elevated: {
    label: 'Elevated',
    color: 'bg-yellow-500',
    position: '37.5%',
    description: 'Slightly above normal. Monitor and maintain a healthy lifestyle.',
  },
  high1: {
    label: 'High (Stage 1)',
    color: 'bg-warning',
    position: '62.5%',
    description: 'Stage 1 hypertension. Consider lifestyle changes and consult a doctor.',
  },
  high2: {
    label: 'High (Stage 2)',
    color: 'bg-destructive',
    position: '87.5%',
    description: 'Stage 2 hypertension. Medical consultation recommended.',
  },
  crisis: {
    label: 'Crisis',
    color: 'bg-red-700',
    position: '95%',
    description: 'Hypertensive crisis. Seek immediate medical attention.',
  },
};

export const HealthInsights = ({
  category,
  pulseAnalysis,
  recommendations,
}: HealthInsightsProps) => {
  const config = categoryConfig[category];

  return (
    <div className="glass-card p-6 animate-fade-in" style={{ animationDelay: '0.2s' }}>
      <div className="flex items-center gap-2 mb-6">
        <Brain className="w-5 h-5 text-primary" />
        <h3 className="text-lg font-display font-semibold">AI Health Insights</h3>
      </div>

      {/* Blood Pressure Category */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm text-muted-foreground">Blood Pressure Category</span>
          <span className={`px-3 py-1 rounded-full text-xs font-medium ${config.color} text-primary-foreground`}>
            {config.label}
          </span>
        </div>
        
        {/* Status bar */}
        <div className="relative">
          <div className="status-bar" />
          <div
            className="absolute top-1/2 -translate-y-1/2 w-4 h-4 rounded-full bg-foreground border-2 border-background shadow-lg transition-all duration-500"
            style={{ left: config.position, transform: `translate(-50%, -50%)` }}
          />
        </div>
        <div className="flex justify-between mt-1 text-[10px] text-muted-foreground">
          <span>Normal</span>
          <span>Elevated</span>
          <span>High 1</span>
          <span>High 2</span>
        </div>
        
        <p className="text-sm text-muted-foreground mt-3">{config.description}</p>
      </div>

      {/* Pulse Analysis */}
      <div className="mb-6 p-4 rounded-lg bg-muted/30 border border-border/50">
        <div className="flex items-center gap-2 mb-2">
          <Activity className="w-4 h-4 text-primary" />
          <span className="text-sm font-medium">Pulse Analysis</span>
        </div>
        <p className="text-sm text-muted-foreground">{pulseAnalysis}</p>
      </div>

      {/* Recommendations */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <TrendingUp className="w-4 h-4 text-primary" />
          <span className="text-sm font-medium">Recommendations</span>
        </div>
        <ul className="space-y-2">
          {recommendations.map((rec, i) => (
            <li key={i} className="flex items-start gap-2 text-sm text-muted-foreground">
              <Heart className="w-3.5 h-3.5 text-primary shrink-0 mt-0.5" />
              <span>{rec}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};
