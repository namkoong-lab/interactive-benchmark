import About from "@/components/home/About";
// import ServiceDemo from "@/components/home/ServiceDemo";
import IdealTrajectory from "@/components/home/IdealTrajectory";
import Citation from "@/components/home/Citation";
import QuickStart from "@/components/home/QuickStart";
// import Contact from "@/components/home/Contact";

export default function Home() {
  return (
    <div className="w-full">
      <About />
      {/* <ServiceDemo /> */}
      <IdealTrajectory />
      <QuickStart />
      <Citation />
      {/* <Contact /> */}
    </div>
  );
}
