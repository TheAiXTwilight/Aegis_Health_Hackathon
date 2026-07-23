import { Routes, Route } from "react-router-dom";
import MainLayout from "../layouts/MainLayout/MainLayout";
import HomePage from "../features/home/HomePage";
import HealthScanPage from "../features/health-scan/HealthScanPage";
import MedicalFormPage from "../features/medical-form/MedicalFormPage";
import ReportPage from "../features/report/ReportPage";
import DashboardRouter from "../features/dashboard/DashboardRouter";
import AboutPage from "../features/about/AboutPage";
import LoginPage from "../features/auth/LoginPage";
import RegisterPage from "../features/auth/RegisterPage";
import ResetPasswordPage from "../features/auth/ResetPasswordPage";
import ProfileSetupPage from "../components/Profile/ProfileSetupPage";
import ProtectedRoute from "../components/ProtectedRoute/ProtectedRoute";

export default function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<MainLayout><HomePage /></MainLayout>} />
      <Route path="/scan" element={
        <MainLayout>
          <ProtectedRoute>
            <HealthScanPage />
          </ProtectedRoute>
        </MainLayout>
      } />
      <Route path="/profile/setup" element={
        <MainLayout>
          <ProtectedRoute>
            <ProfileSetupPage />
          </ProtectedRoute>
        </MainLayout>
      } />
      <Route path="/medical-form" element={
        <MainLayout>
          <ProtectedRoute>
            <MedicalFormPage />
          </ProtectedRoute>
        </MainLayout>
      } />
      <Route path="/report" element={
        <MainLayout>
          <ProtectedRoute>
            <ReportPage />
          </ProtectedRoute>
        </MainLayout>
      } />
      <Route path="/dashboard" element={
        <MainLayout>
          <ProtectedRoute>
            <DashboardRouter />
          </ProtectedRoute>
        </MainLayout>
      } />
      <Route path="/about" element={<MainLayout><AboutPage /></MainLayout>} />
      <Route path="/login" element={<MainLayout><LoginPage /></MainLayout>} />
      <Route path="/register" element={<MainLayout><RegisterPage /></MainLayout>} />
      <Route path="/reset-password" element={<MainLayout><ResetPasswordPage /></MainLayout>} />
    </Routes>
  );
}
