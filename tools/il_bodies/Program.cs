// il_bodies - every method body's IL in a .NET assembly, hashed: "<Type>::<Method> <sha1 of the IL bytes>" one per line, the
// same lines il_bodies.ps1 prints. The closed loop's L19 EQUIVALENCE (Trivial Compiler Equivalence, Papadakis et al. 2015): a
// mutant whose method compiles to the same IL as an unchanged copy of it IS the same program - not a survivor. Reads the file
// with System.Reflection.Metadata (no load of the assembly's code). Only the IL stream is hashed: the local signature lives in the
// method header, so two identical bodies hash alike.
//   dotnet il_bodies.dll <assembly.dll> [<name regex>]
using System;
using System.IO;
using System.Reflection.Metadata;
using System.Reflection.PortableExecutable;
using System.Security.Cryptography;
using System.Text.RegularExpressions;

static class IlBodies
{
    static int Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.Error.WriteLine("usage: il_bodies <assembly.dll> [<name regex>]");
            return 2;
        }
        var match = new Regex(args.Length > 1 ? args[1] : @"__(m\d+|ctl)$");
        using var fs = File.OpenRead(args[0]);
        using var pe = new PEReader(fs);
        var md = pe.GetMetadataReader();
        using var sha = SHA1.Create();
        foreach (var th in md.TypeDefinitions)
        {
            var td = md.GetTypeDefinition(th);
            string tn = md.GetString(td.Name);
            foreach (var mh in td.GetMethods())
            {
                var m = md.GetMethodDefinition(mh);
                string name = md.GetString(m.Name);
                if (!match.IsMatch(name) || m.RelativeVirtualAddress == 0) continue;
                byte[] il = pe.GetMethodBody(m.RelativeVirtualAddress).GetILBytes();
                Console.WriteLine("{0}::{1} {2}", tn, name, Convert.ToHexString(sha.ComputeHash(il)).ToLowerInvariant());
            }
        }
        return 0;
    }
}
