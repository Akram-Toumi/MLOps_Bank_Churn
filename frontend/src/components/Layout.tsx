import React from 'react'

interface LayoutProps {
  children: React.ReactNode
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#f5f5f5' }}>
      <header style={{
        backgroundColor: '#2c3e50',
        color: 'white',
        padding: '1rem 2rem',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
      }}>
        <h2 style={{ margin: 0 }}>Bank Churn Prediction System</h2>
      </header>
      <main style={{ padding: '2rem 0' }}>
        {children}
      </main>
      <footer style={{
        backgroundColor: '#34495e',
        color: 'white',
        padding: '1rem',
        textAlign: 'center',
        marginTop: 'auto'
      }}>
        <p style={{ margin: 0 }}>© 2025 Bank Churn Prediction API</p>
      </footer>
    </div>
  )
}

export default Layout

