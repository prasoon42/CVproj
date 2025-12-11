import { Heart, Droplets, Activity } from 'lucide-react';
import { MetricCard, MetricStatus } from './MetricCard';
import { HealthInsights, BPCategory } from './HealthInsights';
import { ReadingHistory } from './ReadingHistory';

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

interface ResultsDisplayProps {
  result: AnalysisResult;
  history: Reading[];
}

const getSysStatus = (value: number | null): MetricStatus => {
  if (value === null) return 'unknown';
  if (value < 120) return 'normal';
  if (value < 130) return 'elevated';
  return 'high';
};

const getDiaStatus = (value: number | null): MetricStatus => {
  if (value === null) return 'unknown';
  if (value < 80) return 'normal';
  if (value < 90) return 'elevated';
  return 'high';
};

const getPulseStatus = (value: number | null): MetricStatus => {
  if (value === null) return 'unknown';
  if (value >= 60 && value <= 100) return 'normal';
  if (value > 100 || value < 60) return 'elevated';
  return 'normal';
};

const getBPCategory = (sys: number | null, dia: number | null): BPCategory => {
  if (sys === null || dia === null) return 'normal';
  if (sys >= 180 || dia >= 120) return 'crisis';
  if (sys >= 140 || dia >= 90) return 'high2';
  if (sys >= 130 || dia >= 80) return 'high1';
  if (sys >= 120 && dia < 80) return 'elevated';
  return 'normal';
};

const getPulseAnalysis = (pulse: number | null): string => {
  if (pulse === null) return 'Unable to analyze pulse data.';
  if (pulse < 60) return 'Your resting heart rate is below the normal range (bradycardia). This can be normal for athletes, but consult a doctor if you experience symptoms.';
  if (pulse <= 100) return 'Your resting heart rate is within the normal range (60-100 bpm). This indicates good cardiovascular health.';
  return 'Your resting heart rate is above normal (tachycardia). Consider reducing caffeine, managing stress, and consulting a healthcare provider.';
};

const getRecommendations = (category: BPCategory): string[] => {
  const base = [
    'Maintain a balanced diet low in sodium and rich in potassium',
    'Stay physically active with at least 150 minutes of moderate exercise weekly',
  ];

  switch (category) {
    case 'normal':
      return [...base, 'Continue monitoring your blood pressure regularly'];
    case 'elevated':
      return [...base, 'Limit alcohol consumption and quit smoking if applicable', 'Consider stress management techniques'];
    case 'high1':
      return [...base, 'Schedule a consultation with your healthcare provider', 'Monitor blood pressure more frequently'];
    case 'high2':
    case 'crisis':
      return ['Seek medical attention promptly', 'Take prescribed medications as directed', 'Avoid strenuous activities until cleared by a doctor'];
    default:
      return base;
  }
};

export const ResultsDisplay = ({ result, history }: ResultsDisplayProps) => {
  const category = getBPCategory(result.sys, result.dia);

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <MetricCard
          label="Systolic"
          value={result.sys}
          unit="mmHg"
          normalRange="< 120"
          status={getSysStatus(result.sys)}
          icon={<Heart className="w-4 h-4" />}
        />
        <MetricCard
          label="Diastolic"
          value={result.dia}
          unit="mmHg"
          normalRange="< 80"
          status={getDiaStatus(result.dia)}
          icon={<Droplets className="w-4 h-4" />}
        />
        <MetricCard
          label="Pulse Rate"
          value={result.pulse}
          unit="bpm"
          normalRange="60-100"
          status={getPulseStatus(result.pulse)}
          icon={<Activity className="w-4 h-4" />}
        />
      </div>

      {/* Insights and History */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <HealthInsights
          category={category}
          pulseAnalysis={getPulseAnalysis(result.pulse)}
          recommendations={getRecommendations(category)}
        />
        <ReadingHistory readings={history} />
      </div>
    </div>
  );
};
