import { ReactNode } from 'react';
import { CheckCircle2, AlertTriangle, XCircle, HelpCircle } from 'lucide-react';

export type MetricStatus = 'normal' | 'elevated' | 'high' | 'unknown';

interface MetricCardProps {
  label: string;
  value: number | null;
  unit: string;
  normalRange: string;
  status: MetricStatus;
  icon?: ReactNode;
}

const statusConfig = {
  normal: {
    color: 'text-success',
    bgColor: 'bg-success/10',
    borderColor: 'border-success/30',
    icon: CheckCircle2,
    label: 'Normal',
  },
  elevated: {
    color: 'text-warning',
    bgColor: 'bg-warning/10',
    borderColor: 'border-warning/30',
    icon: AlertTriangle,
    label: 'Elevated',
  },
  high: {
    color: 'text-destructive',
    bgColor: 'bg-destructive/10',
    borderColor: 'border-destructive/30',
    icon: XCircle,
    label: 'High',
  },
  unknown: {
    color: 'text-muted-foreground',
    bgColor: 'bg-muted/10',
    borderColor: 'border-muted/30',
    icon: HelpCircle,
    label: 'Unknown',
  },
};

export const MetricCard = ({
  label,
  value,
  unit,
  normalRange,
  status,
  icon,
}: MetricCardProps) => {
  const config = statusConfig[status];
  const StatusIcon = config.icon;

  return (
    <div
      className={`glass-card p-6 border ${config.borderColor} transition-all duration-300 hover:scale-[1.02] animate-fade-in`}
      style={{ animationDelay: '0.1s' }}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          {icon && <span className="text-primary">{icon}</span>}
          <span className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
            {label}
          </span>
        </div>
        <div className={`flex items-center gap-1.5 px-2 py-1 rounded-full ${config.bgColor}`}>
          <StatusIcon className={`w-3.5 h-3.5 ${config.color}`} />
          <span className={`text-xs font-medium ${config.color}`}>{config.label}</span>
        </div>
      </div>

      <div className="flex items-baseline gap-2 mb-3">
        {value !== null ? (
          <>
            <span className={`metric-display ${config.color}`}>{value}</span>
            <span className="text-lg text-muted-foreground font-mono">{unit}</span>
          </>
        ) : (
          <span className="metric-display text-muted-foreground/50">--</span>
        )}
      </div>

      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <span className="px-2 py-0.5 rounded bg-muted/50">Normal: {normalRange}</span>
      </div>
    </div>
  );
};
