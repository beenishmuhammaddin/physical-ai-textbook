import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

// Robot SVG Components
function RobotIcon() {
  return (
    <svg width="120" height="120" viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="robotGrad1" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#667eea" />
          <stop offset="100%" stopColor="#764ba2" />
        </linearGradient>
      </defs>
      <rect x="25" y="35" width="70" height="60" rx="8" fill="url(#robotGrad1)" />
      <circle cx="45" cy="55" r="8" fill="white" />
      <circle cx="75" cy="55" r="8" fill="white" />
      <circle cx="45" cy="55" r="4" fill="#667eea" />
      <circle cx="75" cy="55" r="4" fill="#667eea" />
      <rect x="40" y="72" width="40" height="6" rx="3" fill="white" opacity="0.8" />
      <rect x="15" y="50" width="10" height="30" rx="5" fill="url(#robotGrad1)" />
      <rect x="95" y="50" width="10" height="30" rx="5" fill="url(#robotGrad1)" />
      <rect x="40" y="95" width="12" height="20" rx="4" fill="url(#robotGrad1)" />
      <rect x="68" y="95" width="12" height="20" rx="4" fill="url(#robotGrad1)" />
      <circle cx="60" cy="25" r="8" fill="#f093fb" />
      <rect x="58" y="25" width="4" height="10" fill="#f093fb" />
    </svg>
  );
}

function BrainIcon() {
  return (
    <svg width="120" height="120" viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="brainGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#f093fb" />
          <stop offset="100%" stopColor="#f5576c" />
        </linearGradient>
      </defs>
      <ellipse cx="60" cy="60" rx="35" ry="40" fill="url(#brainGrad)" />
      <path d="M 40 45 Q 50 35 60 45 Q 70 35 80 45" stroke="white" strokeWidth="3" fill="none" opacity="0.6" />
      <path d="M 40 60 Q 50 50 60 60 Q 70 50 80 60" stroke="white" strokeWidth="3" fill="none" opacity="0.6" />
      <path d="M 40 75 Q 50 65 60 75 Q 70 65 80 75" stroke="white" strokeWidth="3" fill="none" opacity="0.6" />
      <circle cx="45" cy="50" r="3" fill="white" />
      <circle cx="60" cy="55" r="3" fill="white" />
      <circle cx="75" cy="50" r="3" fill="white" />
      <circle cx="50" cy="70" r="3" fill="white" />
      <circle cx="70" cy="70" r="3" fill="white" />
    </svg>
  );
}

function GearIcon() {
  return (
    <svg width="120" height="120" viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="gearGrad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#4facfe" />
          <stop offset="100%" stopColor="#00f2fe" />
        </linearGradient>
      </defs>
      <circle cx="60" cy="60" r="18" fill="url(#gearGrad)" />
      <circle cx="60" cy="60" r="10" fill="white" />
      {[0, 45, 90, 135, 180, 225, 270, 315].map((angle, i) => {
        const rad = (angle * Math.PI) / 180;
        const x = 60 + Math.cos(rad) * 28;
        const y = 60 + Math.sin(rad) * 28;
        return (
          <rect
            key={i}
            x={x - 4}
            y={y - 8}
            width="8"
            height="16"
            rx="2"
            fill="url(#gearGrad)"
            transform={`rotate(${angle} ${x} ${y})`}
          />
        );
      })}
      <circle cx="60" cy="60" r="6" fill="#4facfe" />
    </svg>
  );
}

const FeatureList = [
  {
    title: 'Comprehensive Robotics Knowledge',
    IconComponent: RobotIcon,
    color: 'purple',
    description: (
      <>
        Explore the complete spectrum of robotics from fundamental concepts to advanced
        humanoid systems, sensors, actuators, and intelligent control mechanisms.
      </>
    ),
  },
  {
    title: 'AI-Powered Physical Systems',
    IconComponent: BrainIcon,
    color: 'pink',
    description: (
      <>
        Master the integration of artificial intelligence with physical systems,
        learning cutting-edge techniques in machine learning, computer vision, and autonomous decision-making.
      </>
    ),
  },
  {
    title: 'Real-World Applications',
    IconComponent: GearIcon,
    color: 'blue',
    description: (
      <>
        Discover practical applications across industries including manufacturing,
        healthcare, autonomous vehicles, and human-robot collaboration scenarios.
      </>
    ),
  },
];

function Feature({IconComponent, title, description, color}) {
  return (
    <div className={clsx('col col--4')}>
      <div className={clsx(styles.featureCard, styles[`featureCard--${color}`])}>
        <div className={styles.featureIcon}>
          <IconComponent />
        </div>
        <Heading as="h3" className={styles.featureTitle}>{title}</Heading>
        <p className={styles.featureDescription}>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2" className={styles.sectionTitle}>
            Master Physical AI & Robotics
          </Heading>
          <p className={styles.sectionSubtitle}>
            Your comprehensive guide to intelligent physical systems and next-generation robotics
          </p>
        </div>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
