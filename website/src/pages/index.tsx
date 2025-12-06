import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import Chatbot from '../components/Chatbot/Chatbot';
import FloatingChatbot from '../components/FloatingChatbot/FloatingChatbot';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero', styles.heroBanner)}>
      <div className="container">
        <div className={styles.heroContent}>
          <Heading as="h1" className="hero__title fade-in-up">
            {siteConfig.title}
          </Heading>
          <p className="hero__subtitle fade-in-up" style={{animationDelay: '0.2s'}}>
            {siteConfig.tagline}
          </p>
          <div className={clsx(styles.buttons, 'fade-in-up')} style={{animationDelay: '0.4s'}}>
            <Link
              className="button button--primary button--lg"
              to="/docs/introduction-to-physical-ai">
              📚 Start Reading Now
            </Link>
            <Link
              className="button button--secondary button--lg"
              to="#chapters-section"
              onClick={(e) => {
                e.preventDefault();
                const element = document.getElementById('chapters-section');
                if (element) {
                  element.scrollIntoView({ behavior: 'smooth' });
                }
              }}>
              📖 Browse Chapters
            </Link>
          </div>
        </div>

        {/* Animated background elements */}
        <div className={styles.heroBackground}>
          <div className={clsx(styles.backgroundElement, styles.element1)}></div>
          <div className={clsx(styles.backgroundElement, styles.element2)}></div>
          <div className={clsx(styles.backgroundElement, styles.element3)}></div>
        </div>
      </div>
    </header>
  );
}

function HomepageChapters() {
  const chapters = [
    {
      id: '01-introduction-to-physical-ai',
      title: 'Introduction to Physical AI',
      description: 'Explore the fundamentals of embodied artificial intelligence and its applications in robotics.',
      icon: '🤖',
      color: 'blue',
      link: '/docs/introduction-to-physical-ai'
    },
    {
      id: '02-basics-of-humanoid-robotics',
      title: 'Basics of Humanoid Robotics',
      description: 'Learn about the anatomy, kinematics, and dynamics of humanoid robots.',
      icon: '🦾',
      color: 'green',
      link: '/docs/basics-of-humanoid-robotics'
    },
    {
      id: '03-ros2-fundamentals',
      title: 'ROS 2 Fundamentals',
      description: 'Master the Robot Operating System 2 for building robotic applications.',
      icon: '📡',
      color: 'purple',
      link: '/docs/ros2-fundamentals'
    },
    {
      id: '04-digital-twin-simulation',
      title: 'Digital Twin Simulation',
      description: 'Create virtual replicas of robots for safe development and testing.',
      icon: '🎮',
      color: 'orange',
      link: '/docs/digital-twin-simulation'
    },
    {
      id: '05-vision-language-action-systems',
      title: 'Vision-Language-Action Systems',
      description: 'Integrate perception, language understanding, and robotic control.',
      icon: '👁️',
      color: 'teal',
      link: '/docs/vision-language-action-systems'
    },
    {
      id: '06-capstone-ai-robot-pipeline',
      title: 'Capstone AI Robot Pipeline',
      description: 'Build a complete AI-robot system integrating all concepts learned.',
      icon: '🚀',
      color: 'indigo',
      link: '/docs/capstone-ai-robot-pipeline'
    }
  ];

  return (
    <section id="chapters-section" className={clsx('homepage-chapters', styles.chaptersSection)}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <Heading as="h2" className={styles.sectionTitle}>
            📚 Complete Book Chapters
          </Heading>
          <p className={styles.sectionSubtitle}>
            Dive deep into each topic with comprehensive, hands-on content
          </p>
        </div>

        <div className={styles.chaptersGrid}>
          {chapters.map((chapter, index) => (
            <Link
              key={chapter.id}
              to={chapter.link}
              className={clsx(
                styles.chapterCard,
                styles[`chapterCard--${chapter.color}`],
                'fade-in-up'
              )}
              style={{animationDelay: `${index * 0.1}s`}}
            >
              <div className={styles.chapterIcon}>
                {chapter.icon}
              </div>
              <div className={styles.chapterContent}>
                <h3 className={styles.chapterTitle}>
                  Chapter {chapter.id.split('-')[0].replace('0', '')}: {chapter.title}
                </h3>
                <p className={styles.chapterDescription}>
                  {chapter.description}
                </p>
                <div className={styles.chapterMeta}>
                  <span className={styles.chapterId}>Chapter {chapter.id.split('-')[0].replace('0', '')}</span>
                  <span className={styles.readMore}>Read →</span>
                </div>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Welcome to ${siteConfig.title}`}
      description="A comprehensive textbook on Physical AI and Humanoid Robotics">
      <HomepageHeader />
      <main>
        <HomepageChapters />
        <section className={styles.chatbotSection}>
          <div className="container">
            <div className={styles.sectionHeader}>
              <Heading as="h2" className={styles.sectionTitle}>
                🤖 Ask the Robotics AI Assistant
              </Heading>
              <p className={styles.sectionSubtitle}>
                Have questions about Physical AI & Humanoid Robotics? Ask our AI assistant!
              </p>
            </div>
            <div className={styles.chatbotContainer}>
              <Chatbot />
            </div>
          </div>
        </section>
      </main>
      <FloatingChatbot />
    </Layout>
  );
}
