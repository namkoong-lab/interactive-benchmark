import About from "@/components/home/About";
import ServiceDemo from "@/components/home/ServiceDemo";
import RunInstructions from "@/components/home/RunInstructions";
import Citation from "@/components/home/Citation";
import QuickStart from "@/components/home/QuickStart";
import Contact from "@/components/home/Contact";

export default function Home() {
  return (
    <div className="w-full">
      <About />
      <ServiceDemo />
      <Citation />
      <QuickStart />
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <RunInstructions />
      </div>
      <Contact />
    </div>
  );
}
