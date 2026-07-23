import Navbar from "../../components/Navbar/Navbar";
import Footer from "../../components/Footer/Footer";
import "./MainLayout.css";

export default function MainLayout({ children }) {
  return (
    <div className="home-page">
      <div className="page-shell">
        <Navbar />
        <img
          src="/heart.png"
          alt="Heart Layout Graphic"
          className="heart-image"
        />
        <main className="page-content">{children}</main>
        <Footer />
      </div>
    </div>
  );
}