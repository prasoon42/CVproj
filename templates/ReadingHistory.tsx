import { Clock, TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface Reading {
  id: string;
  timestamp: Date;
  sys: number;
  dia: number;
  pulse: number;
}

interface ReadingHistoryProps {
  readings: Reading[];
}

const getTrend = (current: number, previous: number | undefined) => {
  if (!previous) return 'stable';
  if (current > previous + 5) return 'up';
  if (current < previous - 5) return 'down';
  return 'stable';
};

const TrendIcon = ({ trend }: { trend: 'up' | 'down' | 'stable' }) => {
  switch (trend) {
    case 'up':
      return <TrendingUp className="w-3.5 h-3.5 text-destructive" />;
    case 'down':
      return <TrendingDown className="w-3.5 h-3.5 text-success" />;
    default:
      return <Minus className="w-3.5 h-3.5 text-muted-foreground" />;
  }
};

export const ReadingHistory = ({ readings }: ReadingHistoryProps) => {
  if (readings.length === 0) {
    return (
      <div className="glass-card p-6 animate-fade-in" style={{ animationDelay: '0.3s' }}>
        <div className="flex items-center gap-2 mb-4">
          <Clock className="w-5 h-5 text-primary" />
          <h3 className="text-lg font-display font-semibold">Recent Readings</h3>
        </div>
        <div className="text-center py-8 text-muted-foreground">
          <p className="text-sm">No readings yet</p>
          <p className="text-xs mt-1">Upload an image to get started</p>
        </div>
      </div>
    );
  }

  return (
    <div className="glass-card p-6 animate-fade-in" style={{ animationDelay: '0.3s' }}>
      <div className="flex items-center gap-2 mb-4">
        <Clock className="w-5 h-5 text-primary" />
        <h3 className="text-lg font-display font-semibold">Recent Readings</h3>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-border/50">
              <th className="text-left text-xs font-medium text-muted-foreground uppercase tracking-wider py-2">
                Time
              </th>
              <th className="text-center text-xs font-medium text-muted-foreground uppercase tracking-wider py-2">
                SYS
              </th>
              <th className="text-center text-xs font-medium text-muted-foreground uppercase tracking-wider py-2">
                DIA
              </th>
              <th className="text-center text-xs font-medium text-muted-foreground uppercase tracking-wider py-2">
                Pulse
              </th>
            </tr>
          </thead>
          <tbody>
            {readings.map((reading, index) => {
              const prevReading = readings[index + 1];
              return (
                <tr
                  key={reading.id}
                  className="border-b border-border/30 hover:bg-muted/20 transition-colors"
                >
                  <td className="py-3 text-sm text-muted-foreground">
                    {reading.timestamp.toLocaleTimeString([], {
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </td>
                  <td className="py-3 text-center">
                    <div className="flex items-center justify-center gap-1">
                      <span className="font-mono text-sm">{reading.sys}</span>
                      <TrendIcon trend={getTrend(reading.sys, prevReading?.sys)} />
                    </div>
                  </td>
                  <td className="py-3 text-center">
                    <div className="flex items-center justify-center gap-1">
                      <span className="font-mono text-sm">{reading.dia}</span>
                      <TrendIcon trend={getTrend(reading.dia, prevReading?.dia)} />
                    </div>
                  </td>
                  <td className="py-3 text-center">
                    <div className="flex items-center justify-center gap-1">
                      <span className="font-mono text-sm">{reading.pulse}</span>
                      <TrendIcon trend={getTrend(reading.pulse, prevReading?.pulse)} />
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
