import "./About.css";

const AboutPage = () => {
  const teamMembers = [
    {
      name: 'Manpreet',
      role: 'Team Lead',
      contribution : 'Frontend Development, Pipeline & Tool Integration',
      initial: 'M',
      color: '#e0e7ff',
      textColor: '#2563ff',
      description: 'Led the project, developed the frontend, and integrated the core AI pipeline and tools.',
      email: 'manpreetkaursachdev2006@gmail.com',
      linkedin: 'https://www.linkedin.com/in/mksachdevv/',
    },
    {
      name: 'Sahil',
      role: 'Team Member',
      contribution : 'AI Model, Frontend & Backend Development',
      initial: 'S',
      color: '#d1fae5',
      textColor: '#10b981',
      description: 'Built and optimized the AI model, while contributing to frontend and backend development.',
      email: 'aixtwilight@gmail.com',
      linkedin: 'https://www.linkedin.com/in/sahilmkb/',
    },
    {
      name: 'Ayush',
      role: 'Team Member',
      contribution : 'Backend, Deployment & Authentication',
      initial: 'A',
      color: '#f3e8ff',
      textColor: '#a855f7',
      description: 'Contributed to backend, implemented secure authentication, and managed deployment.',
      email: 'ayushkatakwar9011@gmail.com',
      linkedin: 'https://www.linkedin.com/in/ayushk17/',
    },
  ];

  // Helper to extract just the username for a cleaner display
  const getCleanLinkedinDisplay = (url) => {
    const username = url.replace(/^(?:https?:\/\/)?(?:www\.)?linkedin\.com\/in\//i, '').replace(/\/$/, '');
    return `linkedin.com/in/${username}`;
  };

  return (
    <div className="about-layout">
      
      {/* Center-aligned Unified Master Card */}
      <div className="about-master-card">
        
        {/* Header Section */}
        <div className="about-header-section">
          <h1 className="about-main-title">About <span className="brand-highlight">AegisHealth</span></h1>
          {/* UPDATED: Exactly calibrated to span 1.5 to 2 lines across the master card width */}
          <p className="about-description">
            AegisHealth is an advanced, privacy-first medical AI co-pilot engineered to provide institutional-grade clinical intelligence directly on your device. By integrating multi-modal diagnostic processing—spanning automated lab analysis, computer vision X-rays, and drug interaction checks—your sensitive health data remains completely secure and confidential.
          </p>
        </div>

        {/* Symmetrical Upper Divider */}
        <div className="about-section-divider"></div>

        {/* Team Section */}
        <div className="about-team-section">
          <h2 className="section-title">Meet the Team</h2>
          <p className="section-subtitle">A passionate group of innovators building the future of private AI healthcare.</p>

          <div className="team-grid">
            {teamMembers.map((member, index) => (
              <div key={index} className="team-card-inner">
                <div className="team-card-header">
                  <div className="team-avatar" style={{ backgroundColor: member.color, color: member.textColor }}>
                    {member.initial}
                  </div>
                  <h3 className="team-name">{member.name}</h3>
                  <span className="team-role" style={{ color: member.textColor }}>{member.role}</span>
                  
                  {/* Contribution Text */}
                  <span className="team-contribution">{member.contribution}</span>
                  
                </div>
                
                <p className="team-description">{member.description}</p>
                
                <div className="team-divider" />
                
                <div className="team-contact">
                  
                  {/* Clickable Email */}
                  <div className="contact-item" title={member.email}>
                    <svg className="contact-icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path>
                      <polyline points="22,6 12,13 2,6"></polyline>
                    </svg>
                    <a href={`mailto:${member.email}`} className="contact-link">
                      {member.email}
                    </a>
                  </div>

                  {/* Clickable LinkedIn */}
                  <div className="contact-item" title={member.linkedin}>
                    <svg className="contact-icon-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path>
                      <rect x="2" y="9" width="4" height="12"></rect>
                      <circle cx="4" cy="4" r="2"></circle>
                    </svg>
                    <a 
                      href={member.linkedin.startsWith('http') ? member.linkedin : `https://${member.linkedin}`} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="contact-link"
                    >
                      {getCleanLinkedinDisplay(member.linkedin)}
                    </a>
                  </div>

                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Lower Divider */}
        <div className="about-section-divider"></div>

        {/* Mission Section */}
        <div className="about-mission-section">
          <h2 className="section-title">Our Mission</h2>
          <p className="mission-description">
            To empower individuals with AI-driven health insights while ensuring complete data privacy and security.
          </p>

          <div className="mission-grid">
            <div className="mission-item">
              <h3 className="mission-card-title">
                <svg className="mission-icon-svg" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"></path></svg>
                Privacy First
              </h3>
              <p className="mission-card-description">Your data never leaves your device.</p>
            </div>

            <div className="mission-item">
              <h3 className="mission-card-title">
                <svg className="mission-icon-svg" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path></svg>
                AI Powered
              </h3>
              <p className="mission-card-description">Advanced AI models for accurate health insights.</p>
            </div>

            <div className="mission-item">
              <h3 className="mission-card-title">
                <svg className="mission-icon-svg" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path></svg>
                Secure
              </h3>
              <p className="mission-card-description">All processing is done locally and securely.</p>
            </div>

            <div className="mission-item">
              <h3 className="mission-card-title">
                <svg className="mission-icon-svg" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"></path></svg>
                Health Focused
              </h3>
              <p className="mission-card-description">Built to help you live a healthier, better life.</p>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default AboutPage;