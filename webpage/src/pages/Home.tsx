import About from "@/components/home/About";
// import ServiceDemo from "@/components/home/ServiceDemo";
import Citation from "@/components/home/Citation";
import QuickStart from "@/components/home/QuickStart";
// import Contact from "@/components/home/Contact";

export default function Home() {
  return (
    <div className="w-full">
      <About />
      {/* <ServiceDemo /> */}
      <QuickStart />
      <Citation />
      {/* <Contact /> */}
    </div>
  );
}
